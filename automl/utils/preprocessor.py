import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessor:
    def __init__(self, fill_method):

        self._fill_method = fill_method

        self._cat_features = None
        self._scaler = StandardScaler()
        self._one_hot_encoder = OneHotEncoder()
        self.train_data_cols = None

    def fit_transform(self, data: pd.DataFrame, cat_features: [str] = None):
        self.train_data_cols = data.columns
        if cat_features:
            self._cat_features = cat_features
        else:
            self._cat_features = self._determine_cat_features(data)

        # Numerical cols transform
        num_data = data.drop(columns=self._cat_features).copy()
        num_data = self._fill_na(num_data)
        num_data = pd.DataFrame(self._scaler.fit_transform(num_data))
        # Categorical cols transform
        cat_data = data[self._cat_features].copy()
        cat_data = self._fill_na_cat(cat_data)
        cat_data = pd.DataFrame(self._one_hot_encoder.fit_transform(cat_data))

        data = self._merge_num_cat_data(data, num_data, cat_data)
        return data

    def transform(self, data: pd.DataFrame):
        if list(self.train_data_cols.sort_values()) != list(data.columns.sort_values()):
            raise Exception('data columns must match with train data columns')

        # Numerical cols transform
        num_data = data.drop(columns=self._cat_features).copy()
        num_data = self._fill_na(num_data)
        num_data = self._scaler.transform(num_data)
        # Categorical cols transform
        cat_data = data[self._cat_features].copy()
        cat_data = self._fill_na_cat(cat_data)
        cat_data = self._one_hot_encoder.transform(cat_data)

        data = self._merge_num_cat_data(data, num_data, cat_data)
        return data

    def _fill_na(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._fill_method == 'mean':
            for col in X.columns:
                mean = X[col].mean()
                X[col] = X[col].fillna(mean)
        elif self._fill_method == 'median':
            for col in X.columns:
                median = X[col].median()
                X[col] = X[col].fillna(median)
        elif self._fill_method == 'ffill':
            X = X.ffill()
        elif self._fill_method == 'bfill':
            X = X.bfill()
        elif self._fill_method == 'interpolate':
            X = X.interpolate()
        return X

    def _fill_na_cat(self, X: pd.DataFrame) -> pd.DataFrame:
        # fill with most frequent value for cat features
        for col in X.columns:
            X[col] = X[col].fillna(X[col].value_counts().idxmax())
        return X

    @staticmethod
    def _determine_cat_features(data):
        # this dtypes supposed to be categorical
        cat_features = list(data.select_dtypes(include=['object', 'category', 'bool']).columns)
        # also add features with low amount of unique values if amount of cols < rows of data
        if len(data.columns) < data.shape[0]:
            for col in list(data.drop(columns=cat_features).columns):
                n_unique = len(data[col].unique())
                if n_unique <= 8:
                    cat_features.append(col)
        return cat_features

    def _merge_num_cat_data(self, data, num_data, cat_data):
        if len(data.columns) > len(self._cat_features) > 0:
            data = pd.merge(num_data, cat_data, left_index=True, right_index=True)
        elif len(self._cat_features) == len(data.columns):
            data = cat_data
        else:
            data = num_data
        return data