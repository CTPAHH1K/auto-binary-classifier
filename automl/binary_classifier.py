import pandas as pd
import numpy as np
from .utils.models import models_list, MODELS_STR_TO_OBJECT
from .utils.metrics import score_func, higher_better, metrics_list
from .utils.preprocessor import Preprocessor

from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed


class BinaryClassifier:
    def __init__(
            self,
            ensemble: bool = False,
            random_state: int = 42,
            n_jobs: int = -1,
            metric: str = 'roc_auc',
            preprocess_data: bool = True,
            test_size: float = 0.2,
            fill_method='mean',
            verbose_training: bool = True,
    ):
        self._ensemble = ensemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._metric = metric
        self._preprocess_data = preprocess_data
        self._verbose_train = verbose_training
        self._fill_method = fill_method
        self._n_jobs = n_jobs
        if test_size:
            assert 1 > test_size > 0
            self._test_size = test_size

        self.cat_features = None
        self.best_model = None
        self.best_model_name = None

        self.models_list = models_list
        self._preprocessor = Preprocessor(fill_method)
        self.models = {}
        self.models_score = {}

    def fit(self,
            X: pd.DataFrame or np.ndarray,
            y: pd.DataFrame or np.ndarray,
            X_test: pd.DataFrame or np.array = None,
            y_test: pd.DataFrame or np.array = None,
            cat_features: [str] = None,
            verbose: bool = True):
        self._verbose_train = verbose
        self.cat_features = cat_features

        X_train, y_train, X_test, y_test = self._check_convert_inputs(X, y, X_test, y_test)
        # data preprocessing
        if self._preprocess_data:
            X_train = self._preprocess_train(X_train, y_train)
            X_test = self._preprocess_test(X_test)

        # Set worst score
        if higher_better[self._metric]:
            best_score = -np.inf
        else:
            best_score = np.inf

        # pool for parallel model training
        if self._verbose_train:
            verb = 100
        else:
            verb = 0
        pool = Parallel(n_jobs=self._n_jobs, verbose=verb, pre_dispatch='all', backend='multiprocessing')

        # Train models
        models = pool(delayed(self._train_model)(X_train, y_train, X_test, y_test, model) for model in self.models_list)
        models_scores = pool(delayed(self._score_model)(model, X_test, y_test) for model in models)
        for i, model in enumerate(self.models_list):
            self.models[model] = models[i]
            self.models_score[model] = models_scores[i]
            # Score model
            if higher_better[self._metric]:
                if self.models_score[model] > best_score:
                    best_score = self.models_score[model]
                    self.best_model_name = model
                    self.best_model = self.models[model]
            else:
                if self.models_score[model] < best_score:
                    best_score = self.models_score[model]
                    self.best_model_name = model
                    self.best_model = self.models[model]

        print(f'Fit done, best model: {self.best_model_name} with score '
              f'{self._metric}: {self.models_score[self.best_model_name]}')

    def _train_model(self, X, y, X_test, y_test, model):
        # TODO make hyperparameters tuning with hyperopt, etc
        model = MODELS_STR_TO_OBJECT[model]
        model.fit(X, y)
        return model

    def _score_model(self, model, X_test, y_test):
        return score_func[self._metric](y_test, model.predict(X_test))

    def _preprocess_train(self, X_train, y_train):
        X_train = self._preprocessor.fit_transform(X_train, y_train, self.cat_features)
        return X_train

    def _preprocess_test(self, X_test):
        X_test = self._preprocessor.transform(X_test)
        return X_test

    def predict(self, X):
        if type(X) is np.ndarray:
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if self._preprocess_data:
            X = self._preprocessor.transform(X)
        return self.best_model.predict(X)

    def predict_proba(self, X):
        if type(X) is np.ndarray:
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if self._preprocess_data:
            X = self._preprocessor.transform(X)
        return self.best_model.predict_proba(X)

    def _check_convert_inputs(self, X, y, X_test, y_test):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if isinstance(y, pd.DataFrame):
            y = y.values.reshape(-1)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=list(range(X.shape[1])))
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.reshape(-1)

        if (not X_test and y_test) or (not y_test and X_test):
            raise Exception('X_test and y_test must be both set or unset')

        elif not X_test and not y_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size,
                                                                random_state=self._random_state)
        else:
            X_train, y_train = X, y

        return X_train, y_train, X_test, y_test
