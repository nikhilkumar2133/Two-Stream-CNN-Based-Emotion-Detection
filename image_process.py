import pandas as pd
import numpy as np
import cv2
import os

# Load the FER2013 dataset
data = pd.read_csv('D:\8th Sem Project\FER-2013\\fer2013.csv')

# Split the data into training and testing sets
X_train, y_train, X_test, y_test = [], [], [], []
for index, row in data.iterrows():
    pixels = [int(pixel) for pixel in row['pixels'].split()]
    img = np.array(pixels).reshape((48, 48))
    img_float32 = np.float32(img)
    # Convert grayscale image to RGB image
    img = cv2.cvtColor(img_float32, cv2.COLOR_GRAY2RGB)
    # Resize the image to 48x48 pixels
    img = cv2.resize(img, (48, 48))
    # Normalize the pixel values to be between 0 and 1
    img = img.astype('float32') / 255.0
    # Add the preprocessed image and label to the appropriate list
    if row['Usage'] == 'Training':
        X_train.append(img)
        y_train.append(row['emotion'])
    else:
        X_test.append(img)
        y_test.append(row['emotion'])

# Convert the data to 3-D matrices
X_train = np.stack(X_train, axis=0)
X_test = np.stack(X_test, axis=0)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Save the data to files
os.makedirs('preprocessed_dataset', exist_ok=True)
np.save('preprocessed_dataset/X_train.npy', X_train)
np.save('preprocessed_dataset/y_train.npy', y_train)
np.save('preprocessed_dataset/X_test.npy', X_test)
np.save('preprocessed_dataset/y_test.npy', y_test)

print('Part-1 Complete')
