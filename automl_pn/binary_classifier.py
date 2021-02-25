import pandas as pd
import numpy as np
from .utils.models import models_list, MODELS_STR_TO_OBJECT
from .utils.metrics import score_func, higher_better
from .utils.preprocessor import Preprocessor

from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


class BinaryClassifier:
    """
    Automated binary classifier automates some parts of basic ml pipeline.
    Includes data preprocessing (scaling, one-hot encoding, feature generation, feature selection),
    training multiple models, choosing best model.
    """

    def __init__(
            self,
            ensemble: bool = False,
            random_state: int = 42,
            n_jobs: int = -1,
            metric: str = 'roc_auc',
            preprocess_data: bool = True,
            test_size: float = 0.2,
            fill_method='mean',
    ):
        """
        Parameters
        ----------
        ensemble: bool
            Not implemented yet, does nothing
        random_state: int
            Random state
        n_jobs: int
            Number of parallel workers
        metric: str
            Which metric to use. Available metrics : 'roc_auc', 'accuracy', 'f1', 'precision', 'recall'
        preprocess_data: bool
            Set True to use built in data preprocessing
        test_size: float
            Test size to use inside if no test data given on fit
        fill_method: str
            Method to fill missing values, only with preprocess_data=True. Available methods: 'mean', 'median', 'ffill',
             'bfill', 'interpolate'
        """
        self._ensemble = ensemble
        self._random_state = random_state
        self._n_jobs = n_jobs
        self._metric = metric
        self._preprocess_data = preprocess_data
        self._verbose_train = True
        self._fill_method = fill_method
        self._n_jobs = n_jobs
        if test_size:
            if not (0 < test_size < 1):
                raise ValueError(' test_size must be > 0 and < 1')
            self._test_size = test_size

        self.cat_features = None
        self.best_model = None
        self.best_model_name = None

        self.models_list = models_list
        self._preprocessor = Preprocessor(fill_method)
        self.models = {}
        self.models_score = {}
        self._trained = False

    def fit(self,
            X: pd.DataFrame or np.ndarray,
            y: pd.DataFrame or np.ndarray,
            X_test: pd.DataFrame or np.array = None,
            y_test: pd.DataFrame or np.array = None,
            cat_features: [str] = None,
            verbose: bool = True):
        """
        Fit method.

        Parameters
        ----------
        X: pd.DataFrame or np.ndarray
            features array
        y: pd.DataFrame or np.ndarray
            targets array
        X_test: pd.DataFrame or np.ndarray
            optional test features array
        y_test: pd.DataFrame or np.ndarray
            optional test targets array
        cat_features: [str]
            optional list of categorical features
        verbose: bool
            Set true to have log during training

        Returns
        -------

        """
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
        self._trained = True

    def _train_model(self, X: pd.DataFrame or np.ndarray,
                     y: pd.DataFrame or np.ndarray,
                     X_test: pd.DataFrame or np.ndarray,
                     y_test: pd.DataFrame or np.ndarray,
                     model: str):
        """
        Return trained given model with given X_train, y_train
        Parameters
        ----------
        X: pd.DataFrame or np.ndarray
        y: pd.DataFrame or np.ndarray
        X_test: pd.DataFrame or np.ndarray
        y_test: pd.DataFrame or np.ndarray
        model: str
            str from models list

        Returns
        -------
        Trained model instance
        """
        # TODO make hyperparameters tuning with hyperopt, etc
        model = MODELS_STR_TO_OBJECT[model]
        model.fit(X, y)
        return model

    def _score_model(self, model, X_test, y_test) -> float:
        """
        Score model against metric
        Parameters
        ----------
        model: Object with predict method
        X_test: pd.DataFrame or np.ndarray
        y_test: pd.DataFrame or np.ndarray

        Returns
        -------
        float
        """
        return score_func[self._metric](y_test, model.predict(X_test))

    def _preprocess_train(self,
                          X_train: pd.DataFrame or np.ndarray,
                          y_train: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        """
        Method to perform preprocessing on traing data
        Parameters
        ----------
        X_train: pd.DataFrame or np.ndarray
        y_train: pd.DataFrame or np.ndarray

        Returns
        -------
        pd.DataFrame
            processed X_train
        """
        X_train = self._preprocessor.fit_transform(X_train, y_train, self.cat_features)
        return X_train

    def _preprocess_test(self, X_test: pd.DataFrame or np.ndarray) -> pd.DataFrame:
        """
        Method to perform preprocessing on test data
        Parameters
        ----------
        X_test: pd.DataFrame or np.ndarray

        Returns
        -------
        pd.DataFrame
            processed X_test
        """
        X_test = self._preprocessor.transform(X_test)
        return X_test

    def predict(self, X: pd.DataFrame or np.ndarray) -> np.ndarray:
        """
        Predict method.
        Parameters
        ----------
        X: pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray
            array with predicted classes
        """
        if not self._trained:
            raise NotImplementedError("First do fit")
        if type(X) is np.ndarray:
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if self._preprocess_data:
            X = self._preprocessor.transform(X)
        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame or np.ndarray) -> np.ndarray:
        """
        Predict proba method.
        Parameters
        ----------
        X: pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray
            array with predicted classes
        """
        if not self._trained:
            raise NotImplementedError("First do fit")
        if type(X) is np.ndarray:
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if self._preprocess_data:
            X = self._preprocessor.transform(X)
        return self.best_model.predict_proba(X)

    def _check_convert_inputs(self, X, y, X_test, y_test):
        """
        Check input types and do conversion if needed.
        Split X, y to X_train, y_train, X_test, y_test if no test data given
        Parameters
        ----------
        X
        y
        X_test
        y_test

        Returns
        -------
        X_train, y_train, X_test, y_test
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=list(range(X.shape[1])))
        if isinstance(y, pd.DataFrame):
            y = y.values.reshape(-1)
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=list(range(X.shape[1])))
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values.reshape(-1)

        if (not X_test and y_test) or (not y_test and X_test):
            raise AttributeError('X_test and y_test must be both set or unset')

        elif not X_test and not y_test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._test_size,
                                                                random_state=self._random_state)
        else:
            X_train, y_train = X, y

        return X_train, y_train, X_test, y_test
