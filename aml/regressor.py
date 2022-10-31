from aml.estimatorbase import PreprocessorMixin, PredictorMixin

import numpy as np
import pandas as pd

import logging
from typing import List, Dict, Tuple, Callable
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR, LinearSVR

from scipy.stats import uniform

class AutoMLRegressor(PreprocessorMixin, PredictorMixin):
    TEST_SIZE = 0.25
    CV = 5  # folds in cross-validation
    SEARCH_ITERATIONS = 100

    def __init__(self, scoring_function: str = '', n_iter: int = 0,
                 try_LR: bool = False, # Linear Reg
                 try_DT: bool = False, # Decision Tree Regressor
                 try_RF: bool = False, # Random Forest Regressor
                 try_GB: bool = False, # Gradient Boosting for regression
                 try_SVC: bool = False, # Epsilon-Support Vector Regression
                 try_LSVC: bool = False, # Linear Support Vector Regression
                 try_HGB: bool = False, # Histogram-based Gradient Boosting Regression Tree
                 try_MLP: bool = False, # Multi-layer Perceptron regressor.
                 ) -> None:
        self.scoring_function = scoring_function
        self.best_models = []  # sklearn Estimator
        self.random_state = 1234
        self.search_iterations = n_iter or self.SEARCH_ITERATIONS
        self.n_jobs = -1

        self.try_LR = try_LR
        self.try_DT = try_DT
        self.try_RF = try_RF
        self.try_GB = try_GB
        self.try_SVC = try_SVC
        self.try_LSVC = try_LSVC
        self.try_HGB = try_HGB
        self.try_MLP = try_MLP

    def fit(self, X: pd.DataFrame, y: pd.Series,
            categorical: List[str] = [],
            numeric: List[str] = [],
            dates: List[str] = []) -> None:
        self.categorical = categorical
        self.dates = dates
        self.numeric = numeric

        preprocessor = self.get_preprocessor()
        # get total number of features after transformation
        all_features = preprocessor.fit_transform(X)
        tot_num_features = all_features.shape[1]

        categorical_features_bool_mask = [f in categorical for f in X.columns] if len(categorical) > 0 else []#fill with boolean mask for native categorical support

        feature_selector = SelectKBest(score_func=f_regression, k='all')

        models_and_params = []

        if self.try_LR:
            models_and_params.append(self.get_LR_w_params())

        if self.try_HGB:
            models_and_params.append(self.get_HGB_regressor_w_params(tot_num_features, categorical_features_bool_mask))

        if self.try_GB:
            models_and_params.append(self.get_GB_regressor_w_params())

        if self.try_DT:
            models_and_params.append(
                self.get_DTR_with_params(tot_num_features))

        if self.try_RF:
            models_and_params.append(self.get_RF_regressor_w_params(tot_num_features))

        if self.try_SVC:
            models_and_params.append(self.get_SVC_regressor_w_params())

        if self.try_LSVC:
            models_and_params.append(self.get_LSVC_regressor_w_params())

        if self.try_MLP:
            models_and_params.append(self.get_MLP_regressor_w_params())

        for regressor, hyperparams in models_and_params:

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('regressor', regressor)
            ])

            # try:
            if 1 == 1:
                logging.info(f"Trying {regressor.__class__} ...")
                search = RandomizedSearchCV(
                    pipeline, param_distributions=hyperparams, n_iter=self.search_iterations,
                    scoring=self.scoring_function, n_jobs=self.n_jobs)
                search.fit(X, y)
                logging.info('Best parameters: %s' % search.best_params_)
                logging.info('Best CV score: %s' % search.best_score_)
                # for more details:
                #logging.info('Cross-validation results')
                # logging.info(search.cv_results_)

                self.best_models.append(
                    (search.best_estimator_, search.best_params_, search.best_score_))

    def predict(self, X) -> pd.Series:
        # y = np.zeros((X.shape[0], len(self.best_models)))
        y_hat = []
        for i, model in enumerate(self.best_models):
            y_hat.append(model.predict(X))
        return y_hat

    def score(self, X_test, y_test):
        scores = []
        for i, model in enumerate(self.best_models):
            scores.append[self.scorer.__call__(model, X_test, y_test)]
        return scores

    def get_LR_w_params(self):
        regressor = LinearRegression()

        hyperparams = {
            'regressor__fit_intercept': [True, False],
            'regressor__normalize': [True, False],
        }
        return regressor, hyperparams

    def get_DTR_with_params(self, total_features):
        regressor = DecisionTreeRegressor(
            random_state=self.random_state
        )
        hyperparams = {
            'preprocessor__numerical__scaler': [None],  # no need for scaling
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': [*range(1, total_features, 5)] + ['all'],
            'regressor__criterion': ["mse", "friedman_mse", "mae", "poisson"],
            'regressor__max_depth': [*range(1, 100), None],
            'regressor__min_samples_split': [0.001, 0.05, 0.01, 0.05, 0.1, None],
            'regressor__max_leaf_nodes': [*range(5, 100, 15), None],
        }
        return regressor, hyperparams

    def get_GB_regressor_w_params(self):
        regressor = GradientBoostingRegressor(random_state=1234)

        hyperparams = {
            'regressor__max_leaf_nodes': [*range(5, 100, 15), None],
            'regressor__max_depth': [*range(1, 100), None],
            'regressor__max_features': ['auto', 'sqrt', 'log2'],
            'regressor__learning_rate': [.1, 1, .05, .01],
            'regressor__loss' : ['ls', 'lad', 'huber', 'quantile'],
            'regressor__n_estimators': [100, 200, 300, 50],
            'regressor__criterion': ['friedman_mse', 'mse', 'mae'],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf' : [1, 5, 20, 100],
        }
        return regressor, hyperparams

    def get_HGB_regressor_w_params(self, total_features, categorical_features=[]):

        # TODO implement native categorical features
        # regressor = HistGradientBoostingRegressor(random_state=1234, categorical_features=categorical_features)
        regressor = HistGradientBoostingRegressor(random_state=1234)

        hyperparams = {
            # 'preprocessor__numerical__scaler': [None],  # no need for scaling
            # 'feature_selector__k': [*range(1, total_features, 5)] + ['all'],
            'regressor__max_iter': [25, 100, 250],
            'regressor__max_leaf_nodes': [*range(5, 100, 15), None],
            'regressor__max_depth': [*range(1, 100), None],
            'regressor__l2_regularization': [0, *range(1, 100)],
            'regressor__min_samples_leaf' : [1, 5, 20, 100],
            'regressor__learning_rate': [.1, 1, .5,],
            'regressor__loss' : ['least_squares', 'least_absolute_deviation', 'poisson']
        }

        # if len(categorical_features) > 0:
        #     hyperparams['preprocessor__categorical__encoder'] = [None] # no need for one-hot encoding, use parameter
        #     hyperparams['feature_selector__k'] = ['all']

        return regressor, hyperparams

    def get_SVC_regressor_w_params(self):
        regressor = SVR(random_state=1234)

        hyperparams = {
            'regressor__C': uniform(loc=0, scale=4),
            'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'regressor__gamma': ['scale', 'auto'],
            'regressor__coef0': [0.0, 0.1, 1],
        }
        return regressor, hyperparams

    def get_LSVC_regressor_w_params(self):
        regressor = LinearSVR(random_state=1234)
        hyperparams = {
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'regressor__C': np.arange(0.1, 1, 0.1)
        }
        return regressor, hyperparams

    def get_RF_regressor_w_params(self, total_features):
        regressor = RandomForestRegressor(random_state=1234)

        hyperparams = {
            'feature_selector__k': [*range(1, total_features, 5)] + ['all'],
            'regressor__n_estimators': range(1, 200),
            'regressor__criterion': ["mse", "mae"],
            'regressor__max_depth': [*range(1, 100, 2), None],
            'regressor__min_samples_leaf': [*range(1, 20, 1), None],
            'regressor__min_samples_split': [0.001, 0.05, 0.01, 0.05, 0.1, None],
            'regressor__max_leaf_nodes': [*range(5, 100, 15), None],
            'regressor__bootstrap': [True, False]}
        return regressor, hyperparams

    def get_MLP_regressor_w_params(self):
        regressor = MLPRegressor(random_state=1234)

        hyperparams = {
            # TODO: provide in ranges
            'regressor__hidden_layer_sizes': [(10), (25), (50), (100), (50, 10)],
            'regressor__solver': {'lbfgs', 'sgd', 'adam'},
            'regressor__activation': {'logistic', 'tanh', 'relu'},
        }
        return regressor, hyperparams

    def get_XX_regressor_w_params(self):
        regressor = None
        hyperparams = dict()
        return regressor, hyperparams