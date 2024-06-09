from flask import Blueprint, request, jsonify
from .image_prediction import ImagePredictionService

main = Blueprint('main', __name__)

prediction_service = ImagePredictionService()

@main.route('/api/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400

    file = request.files['file']
    
    if file.content_type.split('/')[0] != 'image':
        return jsonify({"detail": "Invalid image file"}), 400
    
    try:
        prediction = prediction_service.predict(file)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"detail": str(e)}), 500
