import tensorflow as tf
import keras.backend as K
from data_preparation import *

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (6, 4), input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(64, (6, 4)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator,
                      validation_data=test_generator,
                      steps_per_epoch=40,
                      epochs=10)

model.save('model.h5')