import numpy as np


class Evaluator:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        pass

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts, ignore_idxs=None):
        for index, (lp, lt) in enumerate(zip(predictions, gts)):
            lp_flatten = lp[ignore_idxs[index] < 1] if ignore_idxs is not None else lp.flatten()
            lt_flatten = lt[ignore_idxs[index] < 1] if ignore_idxs is not None else lt.flatten()
            self.hist += self._fast_hist(lp_flatten, lt_flatten)
        pass

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return np.nanmean(iu[1:])

    pass
