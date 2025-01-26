import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from constants import BATCH_SIZE, TF_EPOCHS, LEARNING_RATE, IMG_SIZE, DATA_DIR, MODEL_PATH_TF, MEAN, STD

class SimpleCNN(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')
        self.pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv2 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')
        self.conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(num_classes)

    def call(self, x, training=False):
        x = self.pool(self.conv1(x))  # [16, 112, 112]
        x = self.pool(self.conv2(x))  # [32, 56, 56]
        x = self.pool(self.conv3(x))  # [64, 28, 28]
        x = self.flatten(x)           # [batch_size, 64*28*28]
        x = self.fc1(x)               # [batch_size, 128]
        x = self.fc2(x)               # [batch_size, num_classes]
        return x

def create_datasets():
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir   = os.path.join(DATA_DIR, 'test')

    train_ds = keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        label_mode='int'
    )
    val_ds = keras.preprocessing.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode='int'
    )

    # Data Augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
        layers.RandomZoom(0.1),
    ])

    # Rescaling
    rescale_layer = layers.Rescaling(1./255)

    # Convert normalization constants to tensors and make their shapes broadcastable
    MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    MEAN = tf.reshape(MEAN, [1, 1, 1, 3])
    STD = tf.reshape(STD, [1, 1, 1, 3])

    def normalize(x, y):
        return (x - MEAN) / STD, y

    train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(normalize,
                            num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize,
                        num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds

if __name__ == "__main__":
    train_ds, val_ds = create_datasets()

    model = SimpleCNN(num_classes=2)

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH_TF,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TF_EPOCHS,
        callbacks=[checkpoint_callback]
    )

    model.save(MODEL_PATH_TF)

    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy']

    epochs_range = range(1, TF_EPOCHS + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('loss_acc_plot_tf.png')
    plt.show()

    print("Training completed.")
