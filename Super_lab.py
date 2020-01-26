import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score, KFold
from sklearn import preprocessing
from sklearn.svm import SVC
from time import time
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import os
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

KNN_PARAMS = {
    "method": KNeighborsClassifier,
    "params": {
        "n_neighbors": 10,
        "algorithm": 'auto',
        "metric": 'chebyshev'
    },
    "test_params": {
        "n_neighbors":  range(50)[1:50],
        "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
        "metric": ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }}

TREES_PARAMS = {
    "method": DecisionTreeClassifier,
    "params": {
        "criterion": 'entropy',
        "splitter": 'best',
        "max_features": 'auto'
    },
    "test_params": {
        "criterion": ['gini', 'entropy'],
        "splitter": ['best', 'random'],
        "max_features": ['auto', 'sqrt', 'log2']
    }}


SVC_PARAMS = {
    "method": SVC,
    "params": {
        "kernel": 'linear',
        "gamma": 'auto',
        "degree": 2,
    },
    "test_params": {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        "gamma": ['auto', 'scale'],
        "degree":  range(10)[1:10]
    }}


class Database:
    def __init__(self, path, test_path):
        self.db = self.get_data(path)
        self.processing_db()
        self.short_db = self.get_new_data()
        self.test_data = self.get_test_data(test_path)

    def get_data(self, path):
        return pd.read_csv(path)

    def processing_db(self):
        self.change_goal()

        self.db.fillna(self.db.mean(), inplace=True)

    def review_data(self):
        hist = self.get_hist()
        self.get_features()
        corr = self.get_corr()
        plt.show()

    def get_hist(self):
        hist = self.db.hist(figsize=(15, 8))
        return hist

    def get_features(self):
        features = self.db.shape[0]
        print(f'Количество записей в базе равно: {features}')

    def get_corr(self):
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        corr = self.db.corr()
        heatmap = sns.heatmap(corr, linewidths=0.25, vmax=1.0,
                              square=True, cmap="coolwarm",
                              linecolor='k', annot=True)
        return heatmap

    def delete_data(self, cols, db):
        db_new = db.copy()
        for col in cols:
            db_new = db_new.drop([col], axis=1)
        return db_new

    def change_goal(self):
        change = {'label': {'male': 1, 'female': 0}}  # label = column name
        self.db.replace(change, inplace=True)

    def get_new_data(self):
        delete_data = ['meanfun', 'minfun', 'maxfun', 'mode', 'centroid',
                       'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']
        db_new = self.delete_data(delete_data, self.db)

        return db_new

    def get_test_data(self, path):
        voices = os.listdir(path)
        checks = []

        for voice in voices:
            check = self.get_data(path + voice)

            drop_data = [
                'Unnamed: 0', 'sound.files', 'selec', 'duration',
                'time.median', 'time.Q25', 'time.IQR', 'time.ent',
                'time.Q75', 'entropy', 'startdom', 'enddom', 'dfslope',
                'meanpeakf'
            ]

            check = self.delete_data(drop_data, check)

            check.rename(columns={'freq.Q25': 'Q25', 'freq.Q75': 'Q75', 'freq.median': 'median', 'freq.IQR': 'IQR'},
                         inplace=True)
            check = check.drop(['meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'], axis=1)
            check = [check.values[0]]
            checks.append(check)
        return checks


class Classification:

    def __init__(self, df, goal='label'):
        self.df = df
        train_data, test_data = train_test_split(self.df, 0.8)
        self.X_train = train_data.drop((goal), axis=1)
        self.y_train = train_data[goal]
        self.X_test = test_data.drop((goal), axis=1)
        self.y_test = test_data[goal]
        self.kf = KFold(n_splits=5, shuffle=True)
        self.initialization()

    def initialization(self):
        scaler = MinMaxScaler()
        self.X_train[self.X_train.columns] = scaler.fit_transform(self.X_train[self.X_train.columns])
        self.X_test[self.X_test.columns] = scaler.fit_transform(self.X_test[self.X_test.columns])

    def test_algorithms(self, arg, checks):
            method = arg["method"]
            best_params = arg["params"]
            test_params = arg["test_params"]

            k = 0
            mas = list()
            results = {'acc': [], 'name_par': [], 'number': []}

            for i, test_param in enumerate(test_params):
                name_par = list(test_params.keys())[i]
                for one_test_par in test_params[test_param]:
                    checking_params = best_params.copy()
                    checking_params[test_param] = one_test_par
                    clf = method(**checking_params)
                    name = clf.__class__.__name__
                    print(f'Выполняется {name} при {test_param} = {one_test_par}')
                    start_recur = time()
                    try:
                        m = self.accuracy(clf)
                    except:
                        print('Не выполнено!!!!!!!!!!!!!')
                        continue
                    end_recur = time()
                    print(f'Точность {round(m,4)*100}% за {end_recur-start_recur}\n')
                    mas.append(m)
                m = max(mas)
                indices = [i for i, j in enumerate(mas) if j == m][0]
                results['acc'].append(round(m, 4) * 100)
                results['name_par'].append(name_par)
                results['number'].append(test_params[name_par][indices])
                mas.clear()
                k += 1
            print(f'\nВывод максимальной точности {name}:\n')

            resulting_clf = dict()

            for i in range(k):
                acc = results['acc'][i]
                name_par = results['name_par'][i]
                ind = results['number'][i]
                resulting_clf[results['name_par'][i]] = results['number'][i]
                print(f'При {name_par} = {ind} точность {acc}%')

            clf = method(**resulting_clf)
            clf.fit(self.X_train, self.y_train)

            self.predicts(clf, checks)

    def predicts(self, clf, checks):
        for check in checks:
            print(clf.predict(check))

    def accuracy(self, clf):
        clf.fit(self.X_train, self.y_train)
        scores = cross_val_score(estimator=clf, X=self.X_test, y=self.y_test, cv=self.kf, scoring='accuracy')
        return scores.mean()


def train_test_split(df, train_percent=.7):
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.ix[perm[:train_end]]
    test = df.ix[perm[train_end:]]
    return train, test


if __name__ == '__main__':

    dataset_path = 'DataSet/voice.csv'

    test_path = 'DataSet/voices/'

    dataset = Database(dataset_path, test_path)

    # dataset.review_data()

    classification_original_db = Classification(dataset.db)

    classification_short_db = Classification(dataset.short_db)

    # classification_original_db.test_algorithms(SVC_PARAMS)

    classification_short_db.test_algorithms(KNN_PARAMS, dataset.test_data)
