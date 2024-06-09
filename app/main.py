from flask import Blueprint, request, jsonify
from .image_prediction import ImagePredictionService
import os
from io import BytesIO
from google.cloud import storage


main = Blueprint('main', __name__)

prediction_service = ImagePredictionService()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'c241-ps505-5a12b402c36f.json'
storage_client = storage.Client()

@main.route('/api/predict', methods=['POST'])
def predict_image():
    if 'filename' not in request.json:
        return jsonify({"detail": "No filename provided"}), 400

    filename = request.json['filename']
    
    try:
        image_bucket = storage_client.get_bucket('c241-ps505-bucket')
        img_blob = image_bucket.blob(f'upload_predicts/{filename}')
        img_data = BytesIO(img_blob.download_as_bytes())
    except Exception as e:
        return jsonify({"detail": str(e)}), 400
    
    try:
        prediction = prediction_service.predict(img_data)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500
