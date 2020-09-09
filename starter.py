import pandas as pd
from cloudburst.client.client import CloudburstConnection
AWS_ELB_ADDR = "a4bfa5f55af0a47468db8042bdb7663a-1805231160.us-east-1.elb.amazonaws.com"
MY_IP = "3.219.231.113"
local = False
local_cloud = CloudburstConnection(AWS_ELB_ADDR, MY_IP, local=local)
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
f_data = pd.read_csv('/home/ubuntu/ml-data/adult.data', names=names)
f_train = pd.read_csv('/home/ubuntu/ml-data/adult.data', names=names)
f_test = pd.read_csv('/home/ubuntu/ml-data/adult.test', names=names, skiprows=1)
# f_all = {'data': f_data, 'train': f_train, 'test': f_test}
def mlviz_two(_, a, b, c):
    import numpy as np
    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    import xgboost as xgb
    import json
    from sklearn.datasets.base import Bunch
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC
    data = a
    train = b
    test = c
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    meta = {'target_names': list(data.income.unique()), 'feature_names': list(data.columns), 'categorical_features': {column: list(data[column].unique()) for column in data.columns if data[column].dtype == 'object'}}
    names = meta['feature_names']
    meta['categorical_features'].pop('income')
    dataset = Bunch(data = train[names[:-1]], target = train[names[-1]], data_test = test[names[:-1]], target_test = test[names[-1]], target_names = meta['target_names'], feature_names = meta['feature_names'], categorical_features = meta['categorical_features'], DESCR = "descr")
    # return dataset

    class EncodeCategorical(BaseEstimator, TransformerMixin):
        """
        Encodes a specified list of columns or all columns if None.
        """
        def __init__(self, columns=None):
            self.columns  = columns
            self.encoders = None
        def fit(self, data, target=None):
            """
            Expects a data frame with named columns to encode.
            """
            # Encode all columns if columns is None
            if self.columns is None:
                self.columns = data.columns
            # Fit a label encoder for each column in the data frame
            self.encoders = {column: LabelEncoder().fit(data[column]) for column in self.columns}
            return self
        def transform(self, data):
            """
            Uses the encoders to transform a data frame.
            """
            output = data.copy()
            for column, encoder in self.encoders.items():
                output[column] = encoder.transform(data[column])
            return output

    encoder = EncodeCategorical(dataset.categorical_features.keys())
    dataset.data = encoder.fit_transform(dataset.data)
    dataset.data_test = encoder.fit_transform(dataset.data_test)

    # return dataset

    class ImputeCategorical(BaseEstimator, TransformerMixin):
        """
        Encodes a specified list of columns or all columns if None.
        """
        def __init__(self, columns=None):
            self.columns = columns
            self.imputer = None
        def fit(self, data, target=None):
            """
            Expects a data frame with named columns to impute.
            """
            # Encode all columns if columns is None
            if self.columns is None:
                self.columns = data.columns
            # Fit an imputer for each column in the data frame
            self.imputer = SimpleImputer(missing_values=0, strategy='most_frequent')
            self.imputer.fit(data[self.columns])
            return self
        def transform(self, data):
            """
            Uses the encoders to transform a data frame.
            """
            output = data.copy()
            output[self.columns] = self.imputer.transform(output[self.columns])
            return output

    imputer = ImputeCategorical(['workclass', 'native-country', 'occupation'])
    dataset.data = imputer.fit_transform(dataset.data)
    dataset.data_test = imputer.fit_transform(dataset.data_test)

    X_train = dataset.data
    yencode = LabelEncoder().fit(dataset.target)
    y_train = yencode.transform(dataset.target)

    X_test = dataset.data_test
    y_test = yencode.transform([y.rstrip(".") for y in dataset.target_test])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    def grid_test_xgboost(colsample_tree, subsample, max_depth, min_child_weight, eta):
        # train model
        params = {'objective': 'multi:softprob', 'num_class': 2, 'eval_metric': 'mlogloss', 'max_depth': max_depth, 'min_child_weight': min_child_weight, 'eta':eta, 'subsample': subsample, 'colsample_bytree': colsample_tree}
        model = xgb.train(params, dtrain, evals=[(dtrain, 'train')], verbose_eval=False)

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        return acc

    def grid_test_svm(kernel, gamma, C):
        clf = SVC(kernel=kernel, gamma=gamma, C=C).fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        return accuracy

    # colsample_tree = [1.0]
    # subsample = [1.0]
    # max_depth = [1, 10]
    # min_child_weight = [1, 10]
    # eta = [.9, .3, .01, .005]

    colsample_tree = [1.0]
    subsample = [1.0]
    max_depth = [1]
    min_child_weight = [1]
    eta = [.9]
    val = None

    for i in colsample_tree:
        for j in subsample:
            for k in max_depth:
                for l in min_child_weight:
                    for m in eta:
                        val = grid_test_xgboost(i, j, k, l, m)

    return val



