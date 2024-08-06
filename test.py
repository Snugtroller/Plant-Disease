import cv2
import pickle
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)["model"]

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image):
    """
    Preprocess the captured image to match the input requirements of the model.
    """
    # Convert the image to RGB (OpenCV captures in BGR by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the input size expected by VGG16
    image_resized = cv2.resize(image_rgb, (224, 224))

    # Convert image to array and preprocess for VGG16
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    # Extract features using VGG16
    features = vgg_model.predict(image_array)
    features_flattened = features.flatten().reshape(1, -1)
    
    return features_flattened

def main():
    # Initialize camera
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame
        image_processed = preprocess_image(frame)

        # Predict using the model
        prediction = model.predict(image_processed)
        
        # Display the prediction on the frame
        label = prediction[0]  # Assuming label is directly usable
        cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Leaf Classification', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
