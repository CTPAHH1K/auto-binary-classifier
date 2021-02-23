from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Optional
from utils.models import models_list, MODELS_STR_TO_OBJECT

class BinaryClassifier:
    def __init__(
            self,
            ensemble: bool = False,
            random_state: int = 1,
            n_jobs: Optional[int] = None,
            metric: str = 'RMSE',
            preprocess_data: bool = True,
    ):
        self._ensemble = ensemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._metric = metric
        self._preprocess_data = preprocess_data

        self.cat_features = None
        self.best_model = None

        self.models_list = models_list
        self.models = {}
        self.models_score = {}

    def fit(self,
            X: pd.DataFrame or np.array = None,
            y: pd.DataFrame or np.array = None,
            X_test: pd.DataFrame or np.array = None,
            y_test: pd.DataFrame or np.array = None,
            cat_features: [str] = None,
            verbose: bool = False):

        if X is None or y is None:
            raise ValueError("X and y must be passed in .fit")
        self.cat_features = cat_features
        if self._preprocess_data:
            X = self._preprocess(X, cat_features)

        for model in self.models_list:
            self.models[model] = self._train_model(model)
            self.models_score[model] = self._score_model(self.models[model])

    def _train_model(self, X_train, y_train, x_test, y_test, model):
        model = MODELS_STR_TO_OBJECT(model)
        model.fit(X_train, y_train)
        return model

    def _score_model(self, model):
        return 0

    def preprocess(self, X_train):
        cat = self.cat_features
        return X_train

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self.best_model.predict_proba(X)
