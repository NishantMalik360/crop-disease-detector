import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Prepare the Data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(224, 224), batch_size=32, class_mode='categorical')
validation_generator = test_datagen.flow_from_directory('dataset/valid', target_size=(224, 224), batch_size=32, class_mode='categorical')

# 2. Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(38, activation='softmax') # 38 classes in the Kaggle dataset
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(train_generator, epochs=10, validation_data=validation_generator) # Increase epochs for better accuracy

# 4. Save the Model
model.save('models/crop_disease_model.h5')
print("Model saved to models/crop_disease_model.h5")

# Save class indices for the web app
import json
with open('models/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)