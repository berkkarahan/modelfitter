import threading

from sklearn.base import clone

class _BaseCVFitter:
    def __init__(self, X, y, models_list, cvgen):
        self.x = X
        self.y = y
        self.cvgen = cvgen

        self._threadlist = []
        self._modellist = models_list
        self._train_ind = []
        self._holdout_ind = []
        self._predictions = []
        self._score = []

        self._build_indices()

    def _build_indices(self):
        for i, tpl in enumerate(self.cvgen.split(self.x, self.y)):
            tr_i, ho_i = tpl
            self._train_ind.append(tr_i)
            self._holdout_ind.append(ho_i)

    def _threaded_fit(self, **kwargs):
        for i, tr_i in enumerate(self._train_ind):
            self._threadlist.append(target=self._fit, args=(self._modellist[i], self.x[tr_i], self.y[tr_i]), kwargs=kwargs)

        for t in self._threadlist:
            t.start()

        for t in self._threadlist:
            t.join()

        print("Model fitting complete.")

    @staticmethod
    def _fit(mdl, x, y, **kwargs):
        return mdl.fit(x, y)

    @staticmethod
    def _predict(model_list, x):
        infoldpreds = []
        for m in model_list:
            infoldpreds.append(m.predict(x))
        infoldpreds = np.hstack(infoldpreds)
        return infoldpreds

    @staticmethod
    def _predict_proba(model_list, x):
        infoldpreds = []
        for m in model_list:
            infoldpreds.append(m.predict_proba(x))
        infoldpreds = np.hstack(infoldpreds)
        return infoldpreds

    @property
    def models(self):
        return self._modellist

    def score(self, skscorer):
        for i, ind_tup in enumerate(zip(self._train_ind, self._holdout_ind)):
            m = self._modellist[i]
            self._predictions.append(m.predict(self.x[ind_tup[1]]))
            self._score.append(skscorer(self.y[ind_tup[1]], self._predictions[i]))
            print("Average score for the given scorer for models are: {}".format(np.mean(self._score)))
