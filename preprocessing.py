import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset/train"

# Categories
categories = [
    "Desert",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake"
]

# Image size
img_size = 128

# Lists to store data
data = []
labels = []

# Read images
for category in categories:

    path = os.path.join(dataset_path, category)

    if not os.path.exists(path):
        print(f"Folder not found: {path}")
        continue

    label = categories.index(category)

    for img in os.listdir(path):

        try:
            img_path = os.path.join(path, img)

            image = cv2.imread(img_path)

            if image is None:
                continue

            # Resize
            resized = cv2.resize(
                image,
                (img_size, img_size)
            )

            # Normalize
            normalized = resized / 255.0

            data.append(normalized)
            labels.append(label)

        except Exception as e:
            print("Error loading image:", img_path)
            print(e)

# Convert to arrays
data = np.array(data)
labels = np.array(labels)

# Dataset Information
print("\nDataset Loaded Successfully")
print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

# Class Mapping
print("\nClass Mapping:")
for i, category in enumerate(categories):
    print(f"{i} -> {category}")

# Display Sample Image
if len(data) > 0:

    plt.imshow(data[0])

    plt.title(
        f"Sample Preprocessed Image\nClass: {categories[labels[0]]}"
    )

    plt.axis("off")
    plt.show()

else:
    print("No images found!")