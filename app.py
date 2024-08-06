import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the VGG16 model pre-trained on ImageNet, excluding the top fully-connected layers
vgg16_model = VGG16(weights="imagenet", include_top=False)

def extract_features(img_rgb):
    # Resize image to the size VGG16 expects
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Convert the image to an array and preprocess it for VGG16
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features using VGG16
    features = vgg16_model.predict(img_array)
    return features.flatten()

data = []
labels = []

data_dir = "./Dataset"
for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        features = extract_features(img_rgb)
        
        data.append(features)
        labels.append(dir_)

# Save the data and labels to a pickle file
with open("leaf_data_vgg16.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Leaf detection data with VGG16 features saved successfully!")
