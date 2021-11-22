from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join
import random
import time

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import threading
from multiprocessing.pool import ThreadPool
from itertools import repeat



def thread_clas():
    clfs = {
        'GNB': GaussianNB(),
        'kNN': KNeighborsClassifier(),
    }
    files = [f for f in listdir('./datasets/') if isfile(join('./datasets', f))]
    random.seed(1111)
    n_datasets = 30
    datasets = random.sample(files, n_datasets)
    # print(datasets)
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
    # threads = []
    # for data_id, dataset in enumerate(datasets):
    #     scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))
    #     t = threading.Thread(target=classify, args=(clfs, dataset, rskf))
    #     threads.append(t)
    #     t.start()
    with ThreadPool(n_datasets) as pool:
        pool.starmap(classify, zip(repeat(clfs), datasets, repeat(rskf)))
    # for index, thread in enumerate(threads):
    #     thread.join()
    # np.save('results', scores)

def classify(clfs, dataset, rskf):
    ds = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
    scores = np.zeros((len(clfs), 5 * 2))
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for clf_id, clf_name in enumerate(clfs):
                clf = clone(clfs[clf_name])
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)

    avg_scores = np.average(scores, axis=1)
    print("Klasyfikator: GNB, Plik: {}, Dokadnosc:{}".format(dataset, round(avg_scores[0],5)))
    print("Klasyfikator: KNN, Plik: {}, Dokadnosc:{}".format(dataset, round(avg_scores[1],5)))


def main():
    thread_clas()


if __name__=="__main__":
    main()
