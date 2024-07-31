"""
Dataset from : https://landcover.ai/
Version 1 dataset
                                    
labels:
    0: Unlabeled background 
    1: Buildings
    2: Woodlands
    3: Water
    4: Road
*N.B
    To install the segmentation models library: pip install -U segmentation-models
    
    If you get an error about generic_utils...

    change 
        keras.utils.generic_utils.get_custom_objects().update(custom_objects) 
    to 
        keras.utils.get_custom_objects().update(custom_objects) 
    in 
        .../lib/python3.7/site-packages/efficientnet/__init__.py 
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Define constants
BATCH_SIZE = 8
N_CLASSES = 5
IMG_HEIGHT = 256  # Adjust as needed
IMG_WIDTH = 256   # Adjust as needed
IMG_CHANNELS = 3
BACKBONE = 'resnet34'

# Set up paths
BASE_DIR = "data/data_for_training_testing_val/"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "train", "masks")
VAL_IMG_DIR = os.path.join(BASE_DIR, "val", "images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val", "masks")

# Initialize preprocessing
scaler = MinMaxScaler()
preprocess_input = sm.get_preprocessing(BACKBONE)

def load_data(img_dir, mask_dir):
    images = []
    masks = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        
        # Load and preprocess image
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img)
        img = scaler.fit_transform(img.reshape(-1, IMG_CHANNELS)).reshape(img.shape)
        img = preprocess_input(img)
        
        # Load and preprocess mask
        mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
        mask = img_to_array(mask).squeeze().astype(int)
        mask = to_categorical(mask, N_CLASSES)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load training and validation data
print("Loading training data")
X_train, y_train = load_data(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
print("Loading validation data")
X_val, y_val = load_data(VAL_IMG_DIR, VAL_MASK_DIR)

# Define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', 
                input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                classes=N_CLASSES, activation='softmax')
model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

print(model.summary())

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=25,
    verbose=1,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('landcover_without_augmentation_uNet.hdf5')

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(history.history['iou_score'])
plt.plot(history.history['val_iou_score'])
plt.title('Model IOU')
plt.ylabel('IOU')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Function to plot sample predictions
def plot_sample_predictions(X, y_true, y_pred, n_samples=3):
    for i in range(n_samples):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(X[i])
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # True mask
        ax2.imshow(np.argmax(y_true[i], axis=-1), cmap='jet')
        ax2.set_title('True Mask')
        ax2.axis('off')
        
        # Predicted mask
        ax3.imshow(np.argmax(y_pred[i], axis=-1), cmap='jet')
        ax3.set_title('Predicted Mask')
        ax3.axis('off')
        
        plt.tight_layout()
        plt.show()

# Make predictions on validation set
y_pred = model.predict(X_val)

# Plot sample predictions
plot_sample_predictions(X_val,y_val,y_pred)