from aml.estimatorbase import PreprocessorMixin, PredictorMixin
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
# from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
#from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay

from scipy.stats import uniform

import logging
from typing import List, Dict, Tuple, Callable


class AutoMLClassifier(PreprocessorMixin, PredictorMixin):

    SEARCH_ITERATIONS = 100
    TEST_SIZE = 0.25
    CV = 5  # folds in cross-validation

    def __init__(self, scoring_function: str = '', n_iter: int = 0,
                 try_LR: bool = False,
                 try_DT: bool = False,
                 try_RF: bool = False,
                 try_GB: bool = False,
                 try_KM: bool = False,
                 try_SVC: bool = False,
                 try_LSVC: bool = False,
                 try_HGB: bool = False,
                 try_MLP: bool = False,
                 ) -> None:
        self.scoring_function = scoring_function
        self.best_models = []  # sklearn Estimator
        self.random_state = 1234
        self.search_iterations = n_iter if n_iter > 0 else self.SEARCH_ITERATIONS
        self.n_jobs = -1

        self.try_LR = try_LR
        self.try_DT = try_DT
        self.try_RF = try_RF
        self.try_GB = try_GB
        self.try_KM = try_KM
        self.try_SVC = try_SVC
        self.try_LSVC = try_LSVC
        self.try_HGB = try_HGB
        self.try_MLP = try_MLP

    def fit(self, X: pd.DataFrame, y: pd.Series,
            categorical: List[str] = [],
            numeric: List[str] = [],
            dates: List[str] = []
            ) -> None:
        self.categorical = categorical
        self.dates = dates
        self.numeric = numeric

        self.labels = list(y.unique())
        if len(self.labels) > 2:
            self.multiclass = True  # multiclass classification

        preprocessor = self.get_preprocessor()
        # get total number of features after transformation
        tot_num_features = preprocessor.fit_transform(X).shape[1]

        feature_selector = SelectKBest(f_classif, k='all')

        models_and_params = []

        #scorer = self.get_scorer()

        if self.try_LR:
            models_and_params.append(self.get_LB_classifier_w_params())

        if self.try_HGB:
            models_and_params.append(self.get_HGB_classifier_w_params(tot_num_features))

        if self.try_GB:
            models_and_params.append(self.get_GB_classifier_w_params())

        if self.try_DT:
            models_and_params.append(
                self.get_DT_classifier_w_params(tot_num_features))

        if self.try_RF:
            models_and_params.append(self.get_RF_classifier_w_params())

        if self.try_KM:
            models_and_params.append(self.get_KM_classifier_w_params())

        if self.try_SVC:
            models_and_params.append(self.get_SVC_classifier_w_params())

        if self.try_LSVC:
            models_and_params.append(self.get_LSVC_classifier_w_params())

        if self.try_MLP:
            models_and_params.append(self.get_MLP_classifier_w_params())

        for classifier, hyperparams in models_and_params:

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selector', feature_selector),
                ('classifier', classifier)
            ])

            # try:
            if 1 == 1:
                logging.info("Trying " + str(type(classifier)) + "...")
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

    def get_LB_classifier_w_params(self):
        classifier = LogisticRegression(
            random_state=self.random_state, class_weight='balanced')

        hyperparams = {
            'classifier__C': uniform(loc=0, scale=4),
            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'classifier__class_weight': [{'False': 0.1, 'True': 100}, {'False': 0.1, 'True': 1000}, 'balanced'],
        }
        return classifier, hyperparams

    def get_DT_classifier_w_params(self, total_features):
        classifier = DecisionTreeClassifier(
            random_state=1234, class_weight='balanced')

        hyperparams = {
            'preprocessor__numerical__scaler': [None],  # no need for scaling
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': [*range(1, total_features, 5)] + ['all'],
            'classifier__max_depth': [*range(1, 100), None],
            'classifier__min_samples_split': [0.001, 0.05, 0.01, 0.05, 0.1, None],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_leaf_nodes': [*range(5, 100, 15), None],
        }
        return classifier, hyperparams

    def get_GB_classifier_w_params(self):
        classifier = GradientBoostingClassifier(random_state=1234)

        hyperparams = {
            'classifier__max_leaf_nodes': [*range(5, 100, 15), None],
            'classifier__max_depth': [*range(1, 100), None],
            'classifier__max_features': ['auto', 'sqrt', 'log2'],
        }
        return classifier, hyperparams

    def get_HGB_classifier_w_params(self, total_features):
        classifier = HistGradientBoostingClassifier(random_state=1234)

        # TODO: use categorical_features parameter!

        hyperparams = {
            'feature_selector__k': [*range(1, total_features, 5)] + ['all'],
            'classifier__max_iter': [25, 100, 250],
            'classifier__max_leaf_nodes': [*range(5, 100, 15), None],
            'classifier__max_depth': [*range(1, 100), None],
            'classifier__l2_regularization': [0, *range(1, 100)],
        }
        return classifier, hyperparams

    def get_SVC_classifier_w_params(self):
        classifier = SVC(random_state=1234, class_weight='balanced')

        hyperparams = {
            'classifier__C': uniform(loc=0, scale=4),
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__coef0': [0.0, 0.1, 1],
        }
        return classifier, hyperparams

    def get_LSVC_classifier_w_params(self):
        classifier = LinearSVC(random_state=1234)
        hyperparams = {
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'classifier__C': np.arange(0.1, 1, 0.1)
        }
        return classifier, hyperparams

    def get_RF_classifier_w_params(self):
        classifier = RandomForestClassifier(
            random_state=1234, class_weight='balanced')

        hyperparams = {
            'classifier__n_estimators': range(1, 200),
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__max_depth': [*range(1, 100, 2), None],
            'classifier__min_samples_leaf': [*range(1, 20, 1), None],
            'classifier__min_samples_split': [0.001, 0.05, 0.01, 0.05, 0.1, None],
            'classifier__max_leaf_nodes': [*range(5, 100, 15), None],
            'classifier__bootstrap': [True, False]}
        return classifier, hyperparams

    def get_KM_classifier_w_params(self):
        classifier = KNeighborsClassifier()

        hyperparams = {
            'classifier__n_neighbors': range(1, 20),
            'classifier__p': [1, 2]}
        return classifier, hyperparams

    def get_MLP_classifier_w_params(self):
        classifier = MLPClassifier(random_state=1234)

        hyperparams = {
            # TODO: provide in ranges
            'classifier__hidden_layer_sizes': [(10), (25), (50), (100), (50, 10)],
            'classifier__solver': ['lbfgs', 'sgd', 'adam'],
            'classifier__activation': ['logistic', 'tanh', 'relu'],
        }
        return classifier, hyperparams

    def get_XX_classifier_w_params(self):
        classifier = None
        hyperparams = dict()
        return classifier, hyperparams

    # def get_scorer(self) -> Callable:
    #     if self.scoring_function == 'roc_auc_score':
    #         scorer = make_scorer(roc_auc_score, average='weighted')
    #     elif self.scoring_function == 'f1_score':
    #         scorer = make_scorer(f1_score, average='weighted')
    #     else:  # 'accuracy_score'
    #         scorer = make_scorer(accuracy_score, average='weighted')
    #     return scorer

    # def predict(self, X: pd.DataFrame) -> List[np.ndarray]:
    #     y_hat = []
    #     for model in self.best_models:
    #         y_hat.append(model.predict(X))
    #     return y_hat

    # def predict_proba(self, X: pd.DataFrame) -> List[np.ndarray]:
    #     y_hat = []
    #     for model in self.best_models:
    #         y_hat.append(model.predict_proba(X))
    #     return y_hat

    # def score(self, X: pd.DataFrame, y: pd.Series) -> List[np.ndarray]:
    #     scores = []
    #     for i, model in enumerate(self.best_models):
    #         scores.append(self.scorer.__call__(model, X, y))
    #     return scores

# %%
