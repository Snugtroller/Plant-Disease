import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

def detect_leaves(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    mask = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)

    # Visualize the mask
    # plt.imshow(mask, cmap='gray')
    # plt.title("Mask")
    # plt.show()

    if np.all(mask == 0) or np.all(mask == 255):
        raise ValueError("Mask is either fully black or fully white, check your thresholding.")

    M = cv2.moments(mask)
    
    # Calculate centroid if moments are valid
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    else:
        return None

data = []
labels = []

data_dir = "./Dataset"
for dir_ in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(data_dir, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        detected_leaves = detect_leaves(img_rgb)
        
        if detected_leaves:
            cx, cy = detected_leaves
            data_aux.append(cx)
            data_aux.append(cy)
            
            data.append(data_aux)
            labels.append(dir_)

# Save the data and labels to a pickle file
with open("leaf_data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Leaf detection data saved successfully!")
