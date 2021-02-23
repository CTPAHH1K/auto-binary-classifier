from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from typing import Optional
from .utils.models import models_list, MODELS_STR_TO_OBJECT
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class BinaryClassifier:
    def __init__(
            self,
            ensemble: bool = False,
            random_state: int = 42,
            n_jobs: Optional[int] = None,
            metric: str = 'ROCAUC',
            preprocess_data: bool = True,
            test_size: float = 0.2
    ):
        self._ensemble = ensemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._metric = metric
        self._preprocess_data = preprocess_data
        self._verbose_train = False
        if test_size:
            assert test_size < 1
            assert test_size > 0
            self._test_size = test_size

        self.cat_features = None
        self.best_model = None

        self.models_list = models_list
        self.models = {}
        self.models_score = {}

    def fit(self,
            X: pd.DataFrame or np.array,
            y: pd.DataFrame or np.array,
            X_test: pd.DataFrame or np.array = None,
            y_test: pd.DataFrame or np.array = None,
            cat_features: [str] = None,
            verbose: bool = True):
        self._verbose_train = verbose

        if not X_test:
            assert y_test is None

        if not X_test and not y_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size,
                                                                random_state=self._random_state)
        else:
            X_train, y_train = X, y

        self.cat_features = cat_features
        if self._preprocess_data:
            X_train = self._preprocess_train(X_train)
            X_test = self._preprocess_test(X_test)

        best_score = -10 #TODO depends on metric
        for model in self.models_list:
            self.models[model] = self._train_model(X_train, y_train, X_test, y_test, model)
            self.models_score[model] = self._score_model(self.models[model], X_test, y_test)
            if self.models_score[model] > best_score:
                self.best_model = self.models[model]

    def _train_model(self, X, y, X_test, y_test, model):
        if self._verbose_train:
            print(f'training {model}')
        model = MODELS_STR_TO_OBJECT[model]
        model.fit(X, y)
        return model

    def _score_model(self, model, X_test, y_test):
        # metric = self._metric
        return roc_auc_score(y_test, model.predict(X_test))

    def _preprocess_train(self, X_train):
        # cat = self.cat_features
        return X_train

    def _preprocess_test(self, X_test):
        # cat = self.cat_features
        return X_test

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        return self.best_model.predict_proba(X)
