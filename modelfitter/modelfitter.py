
import threading

from sklearn.base import clone
from ._base import _BaseCVFitter

class CVModelFit(_BaseCVFitter):
    def __init__(self, X, y, model, cvgen):
        modellist = []
        for _ in range(cvgen.n_splits):
            cloned_mdl = clone(model)
            modellist.append(cloned_mdl)
        super().__init__(X=X, y=y, models_list=modellist, cvgen=cvgen)

    def fit(self, **kwargs):
        self._threaded_fit(**kwargs)

    def predict(self, x, type='default'):
        if type == 'default':
            preds = self._predict(self._modellist, x)
        elif type == 'proba':
            preds = self._predict_proba(self._modellist, x)
        return np.mean(preds, axis=1)
