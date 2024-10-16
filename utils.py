from sklearn.base import BaseEstimator, TransformerMixin
import re

class preprocessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.clean_text).apply(self.convert_text)

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
        text = text.lower()  # Convert to lowercase
        return text

    def convert_text(self, text):
        return text
