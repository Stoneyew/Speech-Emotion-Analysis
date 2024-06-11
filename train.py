from tensorflow.keras.utils import to_categorical
import libro as lf
from lstm import LSTM
import numpy as np
import configs
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

def train_model(config) -> None:
    # Train the model

    # Load features preprocessed by preprocess.py
    x_train, x_test, y_train, y_test = lf.load_feature(config, train=True)

    # Build the model
    model = LSTM.make(
        input_shape=x_train.shape[1],
        rnn_size=configs.rnn_size,
        hidden_size=configs.hidden_size,
        dropout=configs.dropout,
        n_classes=configs.nums_labels,
        lr=configs.lr
    )

    # Train the model
    print('----- Start training', config.model, '-----')
    if config.model in ['lstm', 'cnn1d', 'cnn2d']:
        y_train = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)  # Categorical encoding
        unique_labels_train = np.unique(y_train.argmax(axis=1))
        unique_labels_test = np.unique(y_test)
        print("Unique labels in training set:", unique_labels_train)
        print("Unique labels in test set:", unique_labels_test)
        
        model.train(
            x_train, y_train,
            x_test, y_test_cat,
            batch_size=config.batch_size,
            n_epochs=config.epochs
        )
    else:
        model.train(x_train, y_train)
    print('----- End training ', config.model, ' -----')

    # Evaluate the model
    model.evaluate(x_test, y_test)
    # Save the trained model
    model.save(config.checkpoint_path, config.checkpoint_name)

if __name__ == '__main__':
    train_model(configs)
