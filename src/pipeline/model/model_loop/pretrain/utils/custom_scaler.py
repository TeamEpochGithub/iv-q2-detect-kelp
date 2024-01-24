from sklearn.base import BaseEstimator


class CustomStandardScaler(BaseEstimator):

    def __init__(self):
        self.mean_per_channel = None
        self.std_per_channel = None
    
    def fit(self, X, y=None):
        self.mean_per_channel = X.mean(axis=(0,2,3))
        self.std_per_channel = X.std(axis=(0,2,3))
        return self

    def transform(self, X, y=None):
        return (X - self.mean_per_channel.reshape(1,-1,1,1)) / self.std_per_channel.reshape(1,-1,1,1)