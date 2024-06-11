import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from base import BaseModel
import os
import numpy as np
from typing import Optional
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, ff_dim, input_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)
        self.ffn = models.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(input_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            pos=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # apply sin to even indices in the array; 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

def create_transformer_model(input_shape, num_heads=8, ff_dim=128, num_classes=8):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding
    position_encoding = PositionalEncoding(input_shape[0], input_shape[1])
    encoded_inputs = position_encoding(inputs)
    
    # Transformer Encoder Block
    transformer_block = TransformerBlock(num_heads=num_heads, ff_dim=ff_dim, input_dim=input_shape[1])
    x = transformer_block(encoded_inputs)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense Layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output Layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

class TransformerSER(BaseModel):
    """
    Transformer-based Speech Emotion Recognition model class.

    Args:
        input_shape (tuple): Shape of input features.
        num_heads (int): Number of attention heads.
        ff_dim (int): Feed-forward network dimension.
        num_classes (int): Number of emotion classes.
    """

    def __init__(self, model: tf.keras.Model, trained: bool = False) -> None:
        super(TransformerSER, self).__init__(model, trained)
        print(self.model.summary())

    @classmethod
    def make(cls, 
             input_shape: tuple, 
             num_heads= int, 
             ff_dim= int, 
             num_classes= int, 
             learning_rate= float):
        model = create_transformer_model(input_shape, num_heads, ff_dim, num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return cls(model)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, batch_size: int = 32, n_epochs: int = 20) -> None:
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            shuffle=True
        )

        # Plot training history
        self.plot_training_history(history)

        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("The model has not been trained.")
        return np.argmax(self.model.predict(samples), axis=1)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("The model has not been trained.")
        return self.model.predict(samples)

    def save(self, path: str, name: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        h5_save_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(path, name + ".json")
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()
        
        model = tf.keras.models.model_from_json(loaded_model_json)

        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, True)

    def plot_training_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
