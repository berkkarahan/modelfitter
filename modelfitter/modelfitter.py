import threading

import numpy as np
from sklearn.base import clone


class CVModelFit:
    def __init__(self, X, y, model, cvgen):
        self.x = X
        self.y = y
        self.model = model
        self.cvgen = cvgen

        self._threadlist = []
        self._modellist = []
        self._train_ind = []
        self._holdout_ind = []
        self._predictions = []
        self._score = []

    @staticmethod
    def _fit(mdl, x, y):
        return mdl.fit(x, y)

    @staticmethod
    def _predict(model_list, x):
        infoldpreds = []
        for m in model_list:
            infoldpreds.append(m.predict(x))
        infoldpreds = np.hstack(infoldpreds)
        return infoldpreds

    def fit(self):
        for tr_i, ho_i in self.cvgen.split(self.x, self.y):
            self._train_ind.append(tr_i)
            self._holdout_ind.append(ho_i)
            cloned_mdl = clone(self.model)
            self._modellist.append(cloned_mdl)
            self._threadlist.append(threading.Thread(target=self._fit, args=(cloned_mdl, self.x[tr_i], self.y[tr_i],)))

        for t in self._threadlist:
            t.start()

        for t in self._threadlist:
            t.join()

        print("Model fitting complete.")

    def predict(self, x):
        infoldpreds = self._predict(self._modellist, x)
        return np.mean(infoldpreds, axis=1)

    def score(self, skscorer):
        for i, ind_tup in enumerate(zip(self._train_ind, self._holdout_ind)):
            m = self._modellist[i]
            self._predictions.append(m.predict(self.x[ind_tup[1]]))
            self._score.append(skscorer(self.y[ind_tup[1]], self._predictions[i]))
        print("Average score for the given scorer for models are: {}".format(np.mean(self._score)))