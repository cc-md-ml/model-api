from io import BytesIO
from typing import Union
import numpy as np
import tensorflow as tf
from flask import request
from .model_loader import ModelLoader

class ImagePredictionService:
    def __init__(self, model_file):
        self.model = ModelLoader.load_model(model_file)
        self.label_mapping = {
            0: "Basal Cell Carcinoma",
            1: "Melanoma",
            2: "Viral Skin Infections",
            3: "Benign Keratosis",
            4: "Psoriasis, Lichen Planus, and related diseases",
            5: "Melanocytic Nevi",
            6: "Seborrheic Keratoses and other Benign Tumors",
            7: "Fungal Skin Infections",
            8: "Eczema",
            9: "Atopic Dermatitis"
        }

        self.tresh_dict = {
            "Basal Cell Carcinoma": 0.6,
            "Melanoma": 0.6,
            "Viral Skin Infections": 0.8,
            "Benign Keratosis": 0.7,
            "Psoriasis, Lichen Planus, and related diseases": 0.7,
            "Melanocytic Nevi": 0.5,
            "Seborrheic Keratoses and other Benign Tumors": 0.6,
            "Fungal Skin Infections": 0.5,
            "Eczema": 0.8,
            "Atopic Dermatitis":0.8
        }

        self.img_size = 260

    def predict(self, img_data: BytesIO) -> Union[str, dict]:
        img = self.load_and_preprocess_image(img_data)
        pred = self.model.predict(tf.expand_dims(img, axis=0))

        pred_class_encoded = np.argmax(pred)
        pred_label = self.label_mapping[pred_class_encoded]
        pred_class_prob = float(pred[0][pred_class_encoded])
        class_probabilities = {self.label_mapping[i]: float(prob) for i, prob in enumerate(pred[0])}

        if pred_class_prob >= self.tresh_dict.get(pred_label, 1):
            return {"label": pred_label, "probability": pred_class_prob, "classes_probability": class_probabilities}

        return {"message": "Sorry, we don't have the data yet"}

    def load_and_preprocess_image(self, image_data: BytesIO) -> tf.Tensor:
        image_data = image_data.read()
        img = tf.io.decode_image(image_data, channels=3)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        return img
