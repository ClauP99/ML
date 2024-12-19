from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# Custom transformer for Word2Vec
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, vector_size=100):
        self.model = model
        self.vector_size = vector_size

    def _get_average_word2vec(self, corpus, model, vector_size=100):
        vectors = []
        for doc in corpus:
            tokens = doc.split()
            valid_vectors = [model[word] for word in tokens if word in model]
            if valid_vectors:
                vectors.append(np.mean(valid_vectors, axis=0))
            else:
                vectors.append(np.zeros(vector_size))
        return np.array(vectors)

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        return self._get_average_word2vec(X, self.model, self.vector_size)
    
    
