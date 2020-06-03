import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np


class FeatureExtractor:
    def __init__(self):
        base_model = load_model('/app/model/geo_similarity_model_relu.h5')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense').output)
        
        
    def extract(self, img):  
        img = img.resize((224, 224))  
        img = img.convert('RGB')  
        x = image.img_to_array(img) 
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = preprocess_input(x)  # Subtracting avg values for each pixel
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )

        return feature / np.linalg.norm(feature)  # Normalize
