from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class CosineKMeans(BaseEstimator, ClusterMixin):
    """A wrapper for KMeans that normalizes data to use cosine distance."""

    def __init__(self, n_clusters=8, random_state=None, algorithm='lloyd'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.algorithm = algorithm
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            algorithm=self.algorithm,
        )

    def fit(self, X, y=None):
        """Normalize data and fit KMeans."""
        X_normalized = normalize(X)
        self.model.fit(X_normalized)
        return self

    def predict(self, X):
        """Normalize data and predict cluster labels."""
        X_normalized = normalize(X)
        return self.model.predict(X_normalized)

    def fit_predict(self, X, y=None):
        """Normalize data, fit KMeans, and predict cluster labels."""
        X_normalized = normalize(X)
        return self.model.fit_predict(X_normalized)

    @property
    def inertia_(self):
        """Return the inertia of the fitted model."""
        return self.model.inertia_ters
