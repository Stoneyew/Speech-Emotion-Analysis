import os
import pickle
from abc import ABC
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
import joblib
from base_for_mlp import BaseModel

class MLModel(BaseModel, ABC):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLModel, self).__init__(model, trained)

    def save(self, path: str, name: str) -> None:
        
        save_path = os.path.abspath(os.path.join(path, name + '.m'))
        pickle.dump(self.model, open(save_path, "wb"))

    @classmethod
    def load(cls, path: str, name: str):

        model_path = os.path.abspath(os.path.join(path, name + '.m'))
        model = joblib.load(model_path)
        return cls(model, True)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        self.model.fit(x_train, y_train)
        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        
        if not self.trained:
            raise RuntimeError('There is no trained model.')
        return self.model.predict(samples)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        if hasattr(self, 'reshape_input'):
            samples = self.reshape_input(samples)
        return self.model.predict_proba(samples)[0]


class MLP(MLModel):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLP, self).__init__(model, trained)

    @classmethod
    def make(cls, params):
        model = MLPClassifier(**params)
        return cls(model)
