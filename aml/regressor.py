import numpy as np
import pandas as pd


class AutoMLRegressor:
    TEST_SIZE = 0.25
    CV = 5  # folds in cross-validation
    SEARCH_ITERATIONS = 100

    def __init__(self, scoring_function: str = '', n_iter: int = 0) -> None:
        self.scoring_function = scoring_function
        self.best_models = []  # sklearn Estimator
        self.random_state = 1234
        self.search_iterations = n_iter or self.SEARCH_ITERATIONS

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise Exception('Not implemented!')

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
