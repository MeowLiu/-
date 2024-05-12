import numpy as np


class Perceptron:
    def __init__(self) -> None:
        self._w = None
        self._b = None
        self._eta = None

    def __repr__(self) -> str:
        return f"Perceptron(sign({self._w} * X{" + " if self._b >= 0 else " - "}{abs(self._b)}))"

    def __sign(self, x: np.ndarray) -> int:
        return 1 if self._w.dot(x) + self._b >= 0 else -1

    def __helper(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        for x, y in zip(x_train, y_train):
            while y * (self._w.dot(x) + self._b) <= 0:
                self._w = self._w + self._eta * x * y
                self._b = self._b + self._eta * y
                self.__helper(x_train=x_train, y_train=y_train)

        return None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "Perceptron":
        self._w = np.zeros(x_train.shape[1], dtype=int)
        self._b = 0
        self._eta = 1
        self.__helper(x_train, y_train)

        return self

    def predict(self, x_predict: np.ndarray) -> np.ndarray:
        return np.array([self.__sign(x) for x in x_predict])
