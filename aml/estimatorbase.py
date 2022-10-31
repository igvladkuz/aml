from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
import pandas as pd

from typing import List, Dict, Tuple, Callable
import numpy as np


class PreprocessorMixin:
    def get_preprocessor(self):

        number_types = ['int64', 'float64']  # np.number
        categorical_types = ['object', 'category', 'bool']

        # TODO: determine based on data
        if len(self.categorical) == 0:
            categorical = make_column_selector(
                # dtype_include=categorical_types
                dtype_exclude=number_types
            )
        else:
            categorical = self.categorical
        
        if len(self.numeric) == 0:
            numeric = make_column_selector(
                dtype_include=number_types
            )
        else:
            numeric = self.numeric

        #categorical = X.select_dtypes(include=[object, 'category', bool]).columns

        # TODO implement addition of months, weekday, day of month, hour, mminute, second ?
        if len(self.dates) > 0:
            pass

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(dtype='int'))
        ]
        )

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            [('categorical', cat_pipeline, categorical),
             ('numerical', num_pipeline, numeric)],
            remainder='passthrough'
        )
        return preprocessor


class PredictorMixin:
    def predict(self, X: pd.DataFrame) -> List[np.ndarray]:
        y_hat = []
        for model in self.best_models:
            y_hat.append(model.predict(X))
        return y_hat

    def predict_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
        y_hat = []
        for model in self.best_models:
            y_hat.append(model.predict_proba(X))
        return y_hat
