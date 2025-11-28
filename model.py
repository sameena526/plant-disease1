import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# def build_model(num_classes, img_size=(224,224,3), base_trainable=False):
#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=img_size,
#         include_top=False,
#         weights='imagenet'
#     )
#     base_model.trainable = base_trainable

#     inputs = keras.Input(shape=img_size)
#     x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
#     x = base_model(x, training=False)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.2)(x)
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     model = models.Model(inputs, outputs)
    
#     # Compile the model
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model
def build_model(num_classes, img_size=(224, 224, 3), base_trainable=False):
    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size,
        include_top=False,
        weights="imagenet"
    )
    base.trainable = base_trainable

    inputs = keras.Input(shape=img_size)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # ---- FIX ----
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
