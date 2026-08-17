def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_full_lifecycle(client, clean_invoice_bytes):
    # upload
    r = client.post("/invoices", files={"file": ("kpn.txt", clean_invoice_bytes, "text/plain")})
    assert r.status_code == 201
    iid = r.json()["id"]
    assert r.json()["status"] == "received"

    # process -> known supplier KPN B.V. is seeded -> auto_booked
    r = client.post(f"/invoices/{iid}/process")
    assert r.status_code == 200
    assert r.json()["status"] == "auto_booked"

    # detail
    d = client.get(f"/invoices/{iid}").json()
    assert d["supplier_name"] == "KPN B.V."
    assert len(d["fields"]) == 8

    # correction -> feedback stored, status approved
    r = client.post(f"/invoices/{iid}/corrections",
                    json={"field_name": "total", "corrected_value": "1210.00", "corrected_by": "a@b.nl"})
    assert r.status_code == 201
    assert client.get(f"/invoices/{iid}").json()["status"] == "approved"


def test_process_missing_invoice_404(client):
    assert client.post("/invoices/999/process").status_code == 404


def test_metrics_endpoint(client, clean_invoice_bytes):
    iid = client.post("/invoices", files={"file": ("k.txt", clean_invoice_bytes, "text/plain")}).json()["id"]
    client.post(f"/invoices/{iid}/process")
    m = client.get("/metrics").json()
    assert m["total_processed"] == 1
    assert m["automatic_processing_rate"] == 1.0
