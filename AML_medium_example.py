# %%
# After the example on https://towardsdatascience.com/how-to-build-your-own-automl-library-in-python-from-scratch-995940f3fa71
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


class MyAutoMLClassifier:
    # MyAutoMLClassifier's constructor will accept a scoring function that will be used in k-fold CV and the number of iterations of the random search
    def __init__(self,
                 scoring_function='balanced_accuracy',
                 n_iter=50):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self, X, y):
        X_train = X
        y_train = y

# detect the distinct values of the categorical variables in order to apply one-hot encoding
        categorical_values = []

        cat_subset = X_train.select_dtypes(
            include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(
                list(cat_subset.iloc[:, i].dropna().unique()))
# define a pipeline for the numerical variables, that will be cleaned according to a parameter that will be defined later and scaled according to a scaler that we’ll decide in the random search part.
        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

# define a pre-processing pipeline for the categorical variables. This pipeline will clean the blanks using the most frequent value and will one-hot encode them.
        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])
# Everything is finally included in the ColumnTransformer settings, that will perform all the pre-processing part
        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(
                dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(
                dtype_include=['object', 'category', 'bool']))
        ])
# we have to define the ML pipeline, that is built by the pre-processing phase, the feature selection and the model itself.
# We can set the model to LogisticRegression at the moment; it will be changed later by random search
        model_pipeline_steps = []
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(
            ('feature_selector', SelectKBest(f_classif, k='all')))
        model_pipeline_steps.append(('estimator', LogisticRegression()))
        model_pipeline = Pipeline(model_pipeline_steps)
# we can calculate the number of the features (we’ll need it for the feature selection part) and create an empty list that will contain the optimization grid according to the syntax needed by RandomSearchCV
        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = []

# start adding models to our optimization grid
# Logistic regression
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [LogisticRegression()]
        })
# we are creating an object that will change the scaling among RobustScaler, StandardScaler and MinMaxscaler.
# Then will change the cleaning strategy among mean and median and will select the features from 1 to the total number of features with steps of 5.
# Finally, the model itself is set. The random search will check random combinations of this grid, searching for the one that maximizes the performance metrics in cross-validation.
# We can now add other models with their own pre-processing needs and hyperparameters. For example, trees don’t require any scaling, but SVM do.
# We can add as many models as we want, optimizing their hyperparameters in the same grid

# K-nearest neighbors
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [KNeighborsClassifier()],
            'estimator__weights': ['uniform', 'distance'],
            'estimator__n_neighbors': np.arange(1, 20, 1)
        })

        # Random Forest
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [RandomForestClassifier(random_state=0)],
            'estimator__n_estimators': np.arange(5, 500, 10),
            'estimator__criterion': ['gini', 'entropy']
        })

        # Gradient boosting
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [GradientBoostingClassifier(random_state=0)],
            'estimator__n_estimators': np.arange(5, 500, 10),
            'estimator__learning_rate': np.linspace(0.1, 0.9, 20),
        })

        # Decision tree
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [DecisionTreeClassifier(random_state=0)],
            'estimator__criterion': ['gini', 'entropy']
        })

        # Linear SVM
        optimization_grid.append({
            'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [LinearSVC(random_state=0)],
            'estimator__C': np.arange(0.1, 1, 0.1),

        })

# we are searching for the best combination of cleaning strategy, scaling procedure, set of features, model and hyperparameters’ values, everything in the same search procedure.
# This is the core of any AutoML library and can be extended as you want.
# We have now a complete optimization grid, so we can finally apply the random search to find the best pipeline parameters and save the results in some properties of our object.
# The random search will apply a 5-fold cross-validation with the scoring function and the number of iterations we have chosen in the class constructor
        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_


# Inference methods:
# predict()
# predict_proba()


    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X, y=None):
        return self.best_estimator_.predict_proba(X)

# %%


def show_metrics(y_test, y_hat):
    print("Balanced accuracy score", balanced_accuracy_score(y_test, y_hat))
    print("ROC AUC score", roc_auc_score(y_test, y_hat))
    precision = precision_score(y_test, y_hat)
    print("Precision", precision)
    recall = recall_score(y_test, y_hat)
    print("Recall", recall)
    print("F1 score", 2*precision*recall/(precision+recall))


# %%

# Test on a sample dataset

d = load_breast_cancer()
y = d['target']
X = pd.DataFrame(d['data'], columns=d['feature_names'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = MyAutoMLClassifier()
model.fit(X_train, y_train)


# %%
# show metrics for the model
y_hat = model.predict(X_test)
show_metrics(y_test, y_hat)


# %%
est = model.best_estimator_[-1]  # last: classifier
print(est)
pipe = model.best_pipeline
print(pipe)
# %%
# Credit card Churn prediction

cols = ['Attrition_Flag', 'Customer_Age', 'Gender',
        'Dependent_count', 'Education_Level', 'Marital_Status',
        'Income_Category', 'Card_Category', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']

churn_data = pd.read_csv('data/BankChurners.csv', usecols=cols)

label = 'Attrition_Flag'

# %%
y = churn_data[label]
X = churn_data.drop(label, axis=1)

# convert to list
labels = list(y.unique())
y = y.apply(lambda x: labels.index(x))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1234, test_size=0.2)

churn_model = MyAutoMLClassifier(scoring_function='roc_auc', n_iter=50)
churn_model.fit(X_train, y_train)

# %%
show_metrics(y_test, churn_model.predict(X_test))

# %%
print(churn_model.best_estimator_[-1])  # last: classifier
print(churn_model.best_pipeline)

# %%
# functions to introduce some noise into data

# pick a fraction of rows


def make_NaNs(X, frac=.3, cols=[]):

    n_cols = len(cols)

    nrows = X.shape[0]
    np.random.seed(1234)
    mod_ind = np.random.randint(0, nrows-1, size=round(nrows*frac))

    # and remove value of specific or random columns
    columns = X.columns
    for i in mod_ind:

        if n_cols > 0:
            for c in cols:
                X.iloc[i][c] = np.NaN
                #print(f'Set {i} {c} to NaN')
        else:
            c = np.random.choice(columns)
            X.iloc[i][c] = np.NaN
            #print(f'Set {i} {c} to NaN')
    return X


# %%
d = load_breast_cancer()
y = d['target']
X = pd.DataFrame(d['data'], columns=d['feature_names'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

# %%
# let's mess up some data
X_train = make_NaNs(X_train, .5)
X_train.info()


# %%
# train model on messy data
model = MyAutoMLClassifier()
model.fit(X_train, y_train)

y_hat = model.predict(X_test)
show_metrics(y_test, y_hat)


# %%
# make churn data messy and train model on messy data

y = churn_data[label]
X = churn_data.drop(label, axis=1)

# convert to list
labels = list(y.unique())
y = y.apply(lambda x: labels.index(x))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=1234, test_size=0.2)
X_train = make_NaNs(X_train.copy(), 1, ['Dependent_count'])
# X_train.info()
X_train
#  %%
churn_model = MyAutoMLClassifier(scoring_function='roc_auc', n_iter=50)
churn_model.fit(X_train, y_train)
# %%
