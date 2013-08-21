from pickle import dump as pickle_save, load as pickle_load
from numpy.random import randint
from pandas import DataFrame
from copper import Dataset, ModelComparison
from sklearn.ensemble import RandomForestClassifier

class Koala(object):

    data = None
    classifier = None

    _mc = None

    def __init__(self,data=None, target=None, train=False):
        if data is not None:
            self.set_data(data)
            if target is not None:
                self.set_target(target)
                if train:
                    self.train(test_size=0.1)
        return None

    def save(self, path):
        odata = {'data': self.data, 'mc': self._mc, 'classifier': self.classifier}
        with open(path, 'wb') as f:
            pickle_save(odata, f)

    def load(self, path):
        with open(path, 'rb') as f:
            try:
                odata = pickle_load(f)
                self.data = odata['data']
                self._mc = odata['mc']
                self.classifier = odata['classifier']
            except Exception as e:
                print("Invalid data: %s" % (str(e)))

    def set_data(self, df):
        self.data = Dataset(df)
        return None

    def set_target(self, column):
        self.data.role[column] = self.data.TARGET
        return None

    def train(self, **kwargs):
        test_size = 0.1
        if kwargs.get('test_size') is not None:
            test_size = kwargs.get('test_size')
            del kwargs['test_size']

        self._mc = ModelComparison()
        self._mc.train_test_split(self.data, test_size=test_size, random_state=randint(2**16))
        self._mc['RFC'] = RandomForestClassifier(**kwargs)
        self.classifier = self._mc['RFC']
        self._mc.fit()
        return None

    def predict(self, X):
        try:
            predictions = self._mc.le.inverse_transform(self._mc['RFC'].predict(X))
        except AttributeError:
            predictions = self._mc['RFC'].predict(X)
        finally:
            return predictions

    def accuracy_score(self):
        return self._mc.accuracy_score()[0]

    def precision_score(self, **kwargs):
        return self._mc.precision_score(**kwargs)[0]

    def recall_score(self, **kwargs):
        return self._mc.recall_score(**kwargs)[0]

    def f1_score(self, **kwargs):
        return self._mc.f1_score(**kwargs)[0]

    def metric_score(self, metric, **kwargs):
        return self._mc.metric(metric, **kwargs)[0]

    def confusion_matrix(self):
        return self._mc.cm('RFC')

    def feature_importance(self):
        rf_weights = list(self._mc['RFC'].feature_importances_)
        rf_inputs = list(self.data.filter_cols(role=self.data.INPUT))
        return DataFrame(data={'weight':rf_weights},index=rf_inputs).sort('weight', ascending=False)

    def feature_reduction_scores(self, **kwargs):
        ordered_features = list(self.feature_importance().index)[::-1]
        ordered_features.append(self.data.filter_cols(role=self.data.TARGET)[0])

        idx = []
        f1_score = []
        accuracy_score = []

        for i in range(len(ordered_features[:-1])):
            kl = Koala(data=self.data.frame[ordered_features[i:]], target=self.data.filter_cols(role=self.data.TARGET)[0])
            kl.train(**kwargs)
            idx.append(len(ordered_features[i:])-1)
            f1_score.append(kl.f1_score())
            accuracy_score.append(kl.accuracy_score())
        scores = DataFrame(data={'f1_score': f1_score}, index=idx)
        return scores
