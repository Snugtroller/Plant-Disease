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

def is_likely_leaf(image):
    """
    Perform a check to determine if the image contains a significant amount of green,
    which might indicate it is a leaf.
    """
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the percentage of green pixels in the image
    green_percentage = (np.sum(mask) / mask.size) * 100

    # Return True if green percentage is above a more relaxed threshold
    return green_percentage > 3  # Lowered threshold to be less strict

def main():
    # Initialize camera
    cap = cv2.VideoCapture(1)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Check if the frame is likely a leaf
        if is_likely_leaf(frame):
            # Preprocess the frame
            image_processed = preprocess_image(frame)

            # Predict using the model
            prediction = model.predict(image_processed)

            # Optionally, check confidence if model outputs probabilities
            confidence = max(model.predict_proba(image_processed)[0])

            # Display the prediction if confidence is above a threshold
            if confidence > 0.1:  # Lowered confidence threshold
                label = prediction[0]
                cv2.putText(frame, f'Predicted: {label} ({confidence:.2f})', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'No Leaf Detected', (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No Leaf Detected', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

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
