from flask import Blueprint, request, jsonify
from .image_prediction import ImagePredictionService
from io import BytesIO
from google.cloud import storage
from dotenv import load_dotenv

main = Blueprint('main', __name__)

load_dotenv()
storage_client = storage.Client()
bucket = storage_client.bucket('c241-ps505-bucket')

model_blob = bucket.blob('ml_model/final_model2.h5')
model_file = BytesIO(model_blob.download_as_bytes())

prediction_service = ImagePredictionService(model_file)

@main.route('/api/predict', methods=['POST'])
def predict_image():
    if 'filename' not in request.json:
        return jsonify({"detail": "No filename provided"}), 400

    filename = request.json['filename']
    
    try:
        img_blob = bucket.blob(f'upload_predicts/{filename}')
        img_data = BytesIO(img_blob.download_as_bytes())
    except Exception as e:
        return jsonify({"detail": str(e)}), 400
    
    try:
        prediction = prediction_service.predict(img_data)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500
