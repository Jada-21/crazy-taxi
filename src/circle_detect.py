import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Generate dataset
def gen_data(n=1000, size=64):
    X, y = [], []
    for i in range(n):
        img = np.zeros((size, size))
        if np.random.rand() > 0.5:  # Circle present
            cx, cy, r = np.random.uniform(0.2*size, 0.8*size, 3)
            y_, x_ = np.ogrid[:size, :size]
            mask = (x_ - cx)**2 + (y_ - cy)**2 <= r**2
            img[mask] = 1
            y.append(1)
        else:
            y.append(0)
        X.append(img)

        # Save firsxt 5 data points as images
        if i < 5:
            rgb = tf.image.grayscale_to_rgb(tf.expand_dims(img.astype(np.float32), -1))
            tf.keras.utils.save_img(f"sample_{i}.png", rgb)

    return np.array(X).reshape(-1, size, size, 1), np.array(y)

X, y = gen_data(2000)
X = X / 255.0
X_train, X_test = X[:1600], X[1600:]
y_train, y_test = y[:1600], y[1600:]

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoint: Save best model weights
checkpoint_path = "models/best_circle_detector.weights.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

# history = model.fit(X_train, y_train, epochs=40, validation_data=(X_test, y_test))
# Train
history = model.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop],
    verbose=0
)


# Extract errors (loss)
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Generalization error = validation loss (on unseen data)

# Create plot
plt.figure(figsize=(10, 6))
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b-', label='Training Error', linewidth=2)
plt.plot(epochs, val_loss, 'r--', label='Validation Error', linewidth=2)

plt.title('Training, Validation, and Generalization Error', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Binary Cross-Entropy Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save instead of show
error_plot_path = "plots/error_plot.png"
plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
plt.close()  # Close figure to free memory


# Predict on photo (generate sample)
def predict_circle(model, img_path=None):
    if img_path:
        img = tf.keras.utils.load_img(img_path, target_size=(64,64), color_mode='grayscale')
        img = tf.keras.utils.img_to_array(img) / 255.0

    pred = model.predict(img.reshape(1,64,64,1))[0][0]
    return "Circle detected" if pred > 0.5 else "No circle"

test_image_path = '/workspaces/ml_tf/data/shapes.jpeg'

print(predict_circle(model, test_image_path))  # Run for demo


input_shape = (227, 227, 3)

model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape, padding='valid'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    
    layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
    
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), strides=2),
    
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
