from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# Data Generators
# -----------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

# -----------------------------
# Training Dataset
# -----------------------------

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical'
)

# -----------------------------
# Validation Dataset
# -----------------------------

val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical'
)

# -----------------------------
# Print Class Mapping
# -----------------------------

print("\nClass Mapping:")
print(train_data.class_indices)

# -----------------------------
# CNN Model
# -----------------------------

model = Sequential()

# Layer 1
model.add(
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(128,128,3)
    )
)
model.add(MaxPooling2D(2,2))

# Layer 2
model.add(
    Conv2D(
        64,
        (3,3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2,2))

# Layer 3
model.add(
    Conv2D(
        128,
        (3,3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2,2))

# Extra Layer (helps accuracy)
model.add(
    Conv2D(
        256,
        (3,3),
        activation='relu'
    )
)
model.add(MaxPooling2D(2,2))

# Flatten
model.add(Flatten())

# Dense Layer
model.add(
    Dense(
        128,
        activation='relu'
    )
)

# Reduced Dropout
model.add(
    Dropout(
        0.3
    )
)

# Output Layer
model.add(
    Dense(
        train_data.num_classes,
        activation='softmax'
    )
)

# -----------------------------
# Compile Model
# -----------------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Summary
# -----------------------------

model.summary()

# -----------------------------
# Early Stopping
# -----------------------------

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# -----------------------------
# Train Model
# -----------------------------

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[early_stop]
)

# -----------------------------
# Save Model
# -----------------------------

model.save("model.h5")

print("\nModel Trained Successfully!")
print("Model Saved as model.h5")

# -----------------------------
# Test Dataset Evaluation
# -----------------------------

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(128,128),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

loss, accuracy = model.evaluate(test_data)

print("\nTest Accuracy:", accuracy * 100)