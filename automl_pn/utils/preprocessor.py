import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier


class Preprocessor:
    """
    Automated data preprocessor automates some parts of basic ml pipeline.
    Includes data scaling, one-hot encoding, feature generation, feature selection.
    """
    def __init__(self, fill_method):

        self._fill_method = fill_method

        self._cat_features = None
        self._scaler = StandardScaler()
        self._one_hot_encoder = OneHotEncoder()
        self.train_data_cols = None
        self.drop_columns = set()
        self._trained = False

    def fit_transform(self, data: pd.DataFrame, target, cat_features: [str] = None):
        """
        Method to do data preprocessing and learn how to transform test data in future
        Parameters
        ----------
        data
        target
        cat_features

        Returns
        -------

        """
        self.train_data_cols = data.columns
        if cat_features:
            self._cat_features = cat_features
        else:
            self._cat_features = self._determine_cat_features(data)

        # Numerical cols transform
        num_data = data.drop(columns=self._cat_features).copy()
        num_data = self._fill_na(num_data)
        num_data = pd.DataFrame(self._scaler.fit_transform(num_data))
        num_data = self._feature_engineering_num_fit_transform(num_data)
        # Categorical cols transform
        cat_data = data[self._cat_features].copy()
        cat_data = self._fill_na_cat(cat_data)
        cat_data = self._feature_engineering_cat_fit_transform(cat_data)

        data = self._merge_num_cat_data(data, num_data, cat_data)
        # feature selection
        data = self._feature_selection_fit_transform(data, target)
        self._trained = True
        return data

    def transform(self, data: pd.DataFrame):
        """
        Method to transform data.
        Parameters
        ----------
        data

        Returns
        -------

        """
        if list(self.train_data_cols.sort_values()) != list(data.columns.sort_values()):
            raise ValueError('data columns must match with train data columns')
        if not self._trained:
            raise NotImplementedError('First do fit_transform')
        # Numerical cols transform
        num_data = data.drop(columns=self._cat_features).copy()
        num_data = self._fill_na(num_data)
        num_data = pd.DataFrame(self._scaler.transform(num_data))
        num_data = self._feature_engineering_num_transform(num_data)
        # Categorical cols transform
        cat_data = data[self._cat_features].copy()
        cat_data = self._fill_na_cat(cat_data)
        cat_data = self._feature_engineering_cat_transform(cat_data)

        data = self._merge_num_cat_data(data, num_data, cat_data)
        # feature selection
        data = self._feature_selection_transform(data)
        return data

    def _fill_na(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        return dataframe with filled missing values by defined method
        Parameters
        ----------
        X: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
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
        """
        return dataframe with filled missing values with most frequent value
        Parameters
        ----------
        X: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        # fill with most frequent value for cat features
        for col in X.columns:
            X[col] = X[col].fillna(X[col].value_counts().idxmax())
        return X

    @staticmethod
    def _determine_cat_features(data: pd.DataFrame) -> [str]:
        """
        return list of potentialy categorical featured
        Parameters
        ----------
        data: pd.DataFrame

        Returns
        -------
        [str]
        """
        # this dtypes supposed to be categorical
        cat_features = list(data.select_dtypes(include=['object', 'category', 'bool']).columns)
        # also add features with low amount of unique values if amount of cols < rows of data
        if len(data.columns) < data.shape[0]:
            for col in list(data.drop(columns=cat_features).columns):
                n_unique = len(data[col].unique())
                if n_unique <= 8:
                    cat_features.append(col)
        return cat_features

    def _merge_num_cat_data(self, data: pd.DataFrame, num_data: pd.DataFrame, cat_data: pd.DataFrame) -> pd.DataFrame:
        """
        return merged dataframes
        Parameters
        ----------
        data: pd.DataFrame
        num_data: pd.DataFrame
        cat_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        if len(data.columns) > len(self._cat_features) > 0:
            data = pd.merge(num_data, cat_data, left_index=True, right_index=True)
        elif len(self._cat_features) == len(data.columns):
            data = cat_data
        else:
            data = num_data
        return data

    def _feature_engineering_num_fit_transform(self, num_data: pd.DataFrame) -> pd.DataFrame:
        """
        return initial pd.DataFrame with extra features and learn how to make them on train data
        Parameters
        ----------
        num_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return self._feature_engineering_num(num_data)

    def _feature_engineering_num_transform(self, num_data: pd.DataFrame) -> pd.DataFrame:
        """
        return initial pd.DataFrame with extra features
        Parameters
        ----------
        num_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return self._feature_engineering_num(num_data)

    def _feature_engineering_num(self, num_data: pd.DataFrame) -> pd.DataFrame:
        """
        return initial pd.DataFrame with extra features
        Parameters
        ----------
        num_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        # TODO add some smart feature engineering, maybe with some fe lib
        extra_features = pd.DataFrame()
        for col1 in num_data.columns:
            for col2 in num_data.columns:
                extra_features[f'{col1}*{col2}'] = num_data[col1] * num_data[col2]
                if col1 != col2:
                    extra_features[f'{col1}/{col2}'] = num_data[col1] / num_data[col2]
        return pd.merge(num_data, extra_features, left_index=True, right_index=True)

    def _feature_engineering_cat_fit_transform(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """
        return initial pd.DataFrame with extra features and learn how to make them on train data
        Parameters
        ----------
        cat_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(self._one_hot_encoder.fit_transform(cat_data))

    def _feature_engineering_cat_transform(self, cat_data: pd.DataFrame) -> pd.DataFrame:
        """
        return initial pd.DataFrame with extra features
        Parameters
        ----------
        cat_data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(self._one_hot_encoder.transform(cat_data))

    def _feature_selection_fit_transform(self, data: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
        """
        return dataframe with selected features and learn which features to drop
        Parameters
        ----------
        data: pd.DataFrame
        target: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        # identify overcorrelated features
        CORR_THRESHOLD = 0.95
        corr_matrix = data.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column].abs() > CORR_THRESHOLD)]

        self.drop_columns.update(to_drop)
        # train lgbm to identify low inportant features
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, verbose=-1)
        model.fit(data, target)
        feature_importance_values = model.feature_importances_
        feature_importances = pd.DataFrame({'feature': data.columns, 'importance': feature_importance_values})
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
        zero_importance = list(feature_importances[feature_importances['importance'] == 0.0].feature.values)

        self.drop_columns.update(zero_importance)

        feature_importances = feature_importances.sort_values('cumulative_importance')
        low_importance = list(feature_importances[feature_importances['cumulative_importance'] > 0.995].feature.values)

        self.drop_columns.update(low_importance)

        return data.drop(columns=list(self.drop_columns))

    def _feature_selection_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        return dataframe with selected features
        ----------
        data: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        return data.drop(columns=list(self.drop_columns))

