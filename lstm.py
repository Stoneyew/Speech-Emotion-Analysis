from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam

import os
from typing import Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt

def plot_curve(train: list, val: list, title: str, y_label: str) -> None:
    # Plot training and validation curves
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

class BaseModel(ABC):
    # Base class for all models

    def __init__(self, model: Union[Sequential, BaseEstimator], trained: bool = False) -> None:
        self.model = model
        self.trained = trained  # Indicates if the model is trained

    @abstractmethod
    def train(self) -> None:
        # Train the model
        pass

    @abstractmethod
    def predict(self, samples: np.ndarray) -> np.ndarray:
        # Predict the emotion of audio samples
        pass

    @abstractmethod
    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        # Predict the probabilities of each emotion for audio samples
        pass

    @abstractmethod
    def save(self, path: str, name: str) -> None:
        # Save the model
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str, name: str):
        # Load the model
        pass

    @classmethod
    @abstractmethod
    def make(cls):
        # Build the model
        pass

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        # Evaluate the model on test data and print accuracy
        predictions = self.predict(x_test)
        accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
        print('Accuracy: %.3f\n' % accuracy)
        return accuracy

class DNN(BaseModel):
    # Base class for all Keras-based deep learning models

    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())

    def save(self, path: str, name: str) -> None:
        # Save the model to the specified path
        h5_save_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(path, name + ".json")
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        # Load the model from the specified path
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)

        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, True)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, batch_size: int = 32, n_epochs: int = 20) -> None:
        # Train the model
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, shuffle=True, validation_data=(x_val, y_val))

        # Training and validation accuracy and loss
        train_acc = history.history["accuracy"]
        train_loss = history.history["loss"]
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]

        plot_curve(train_acc, val_acc, "Accuracy", "Accuracy")
        plot_curve(train_loss, val_loss, "Loss", "Loss")

        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        # Predict the emotion of audio samples
        if not self.trained:
            raise RuntimeError("No trained model available.")

        samples = self.reshape_input(samples)
        return np.argmax(self.model.predict(samples), axis=1)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        # Predict the probabilities of each emotion for audio samples
        if not self.trained:
            raise RuntimeError("No trained model available.")

        samples = self.reshape_input(samples)
        return self.model.predict(samples)[0]

    @abstractmethod
    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        # Reshape input data for the model
        pass

class LSTM(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(LSTM, self).__init__(model, trained)

    @classmethod
    def make(cls, input_shape: int, rnn_size: int, hidden_size: int, dropout: float, n_classes: int, lr: float):
        # Build the LSTM model
        model = Sequential()
        
        model.add(KERAS_LSTM(rnn_size, input_shape=(1, input_shape)))
        model.add(Dropout(dropout))
        model.add(Dense(hidden_size, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        optimizer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        # Reshape 2D array to 3D for LSTM input
        return np.reshape(data, (data.shape[0], 1, data.shape[1]))
