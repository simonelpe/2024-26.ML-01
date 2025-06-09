from flask import Flask, request, jsonify
import joblib
import pandas as pd
import time

app = Flask(__name__)

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    param1 = data.get('region')
    param2 = data.get('crop_type')
    param3 = data.get('soil_moisture_%')
    param4 = data.get('soil_pH')
    param5 = data.get('temperature_C')
    param6 = data.get('rainfall_mm')
    param7 = data.get('humidity_%')
    param8 = data.get('sunlight_hours')
    param9 = data.get('irrigation_type')
    param10 = data.get('fertilizer_type')
    param11 = data.get('pesticide_usage_ml')
    param12 = data.get('total_days')
    param13 = data.get('NDVI_index')
    param14 = data.get('crop_disease_status')
    param15 = data.get('harvest_date_sin')
    param16 = data.get('harvest_date_cos')
    param17 = data.get('sowing_date_sin')
    param18 = data.get('sowing_date_cos')

    parameters = pd.DataFrame({
        'region': [param1],
        'crop_type': [param2],
        'soil_moisture_%': [param3],
        'soil_pH': [param4],
        'temperature_C': [param5],
        'rainfall_mm': [param6],
        'humidity_%': [param7],
        'sunlight_hours': [param8],
        'irrigation_type': [param9],
        'fertilizer_type': [param10],
        'pesticide_usage_ml': [param11],
        'total_days': [param12],
        'NDVI_index': [param13],
        'crop_disease_status': [param14],
        'harvest_date_sin': [param15],
        'harvest_date_cos': [param16],
        'sowing_date_sin': [param17],
        'sowing_date_cos': [param18]
        })


    modello = joblib.load('best_grid.joblib')
    start = time.perf_counter()
    infer_results = modello.predict(parameters)
    end = time.perf_counter()
    latency = end - start

    response_data = {
        'result': {
            'value': float(infer_results[0])
        },
	    "status": "OK",
	    "stats": {
		    "model_latency":latency
	    }
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
