import pytest
from pe.app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):
    response = client.post("/infer", json={"region": "Central USA",
        "crop_type": "Cotton",
        "soil_moisture_%": 19.37,
        "soil_pH": 5.92,
        "temperature_C": 33.86,
        "rainfall_mm": 269.09,
        "humidity_%": 55.73,
        "sunlight_hours": 7.93,
        "irrigation_type": "None",
        "fertilizer_type": "Mixed",
        "pesticide_usage_ml": 25.65,
        "total_days": 105,
        "NDVI_index": 0.84,
        "crop_disease_status": "Severe",
        "harvest_date_sin": 0.5,
        "harvest_date_cos": -0.866025,
        "sowing_date_sin": 0.866025,
        "sowing_date_cos": 0.5
        })

    assert response.status_code == 200

    data = response.get_json()
    assert data['result']['value'] is not None, "Il valore di ritorno non deve essere None"
    assert isinstance(data['result']['value'], float), "Il valore di ritorno è un float"
    assert data['result']['value'] > 0, "Il valore di ritorno deve essere positivo"
    assert data['result']['value'] < 10000, "Il valore di ritorno deve essere inferiore a 10000"
    assert data['latency'] is not None, "La latenza non deve essere None"
    assert isinstance(data['latency'], float), "La latenza è un float"
    assert data['latency'] > 0, "La latenza deve essere positiva"
