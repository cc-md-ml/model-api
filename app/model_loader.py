import os
from tempfile import NamedTemporaryFile
import tensorflow as tf
import tensorflow_hub as hub
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()
storage_client = storage.Client()

class ModelLoader:
    @staticmethod
    def load_model(model_file):
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(model_file.read())
            tmp_file.flush()
            tmp_model_path = tmp_file.name
        
        try:
            model = tf.keras.models.load_model(
                tmp_model_path,
                custom_objects={'KerasLayer': hub.KerasLayer}
            )
        finally:
            os.remove(tmp_model_path)
        
        return model
    
    
