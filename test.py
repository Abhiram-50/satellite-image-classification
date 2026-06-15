import numpy as np
from PIL import Image
import tensorflow as tf

model = tf.keras.models.load_model(
    "model.h5"
)

classes = [


    "Desert",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake"
]

image_path =  "dataset/train/Forest/Forest_44.jpg"   # replace with actual image

image = Image.open(image_path)

image = image.resize((128,128))

img_array = np.array(image) / 255.0

img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

predicted_class = classes[np.argmax(prediction)]

confidence = np.max(prediction) * 100

print("Predicted Class:", predicted_class)
print("Confidence:", confidence)