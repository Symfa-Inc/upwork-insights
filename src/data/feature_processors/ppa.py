import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class PreProcessingTransformer(BaseEstimator, TransformerMixin):
    """Custom PreProcessing Transformer."""

    def __init__(self, method='normalize', param=None):
        """Initialize the transformer.

        Args:
        method (str): The pre-processing technique to apply (e.g., 'normalize', 'standardize').
        param (dict): Additional parameters for the pre-processing method.
        """
        self.method = method
        self.param = param

    def fit(self, X, y=None):
        """Fit the pre-processing transformer to the data.

        Args:
        X (array-like): The input data to fit.
        y (array-like, optional): The target values (ignored).

        Returns:
        self: Fitted transformer instance.
        """
        if self.method == 'standardize':
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        elif self.method == 'normalize':
            self.max_ = np.max(X, axis=0)
        # Add other methods if needed
        return self

    def transform(self, X):
        """Apply the pre-processing transformation to the data.

        Args:
        X (array-like): The input data to transform.

        Returns:
        array-like: Transformed data.
        """
        if self.method == 'standardize':
            return (X - self.mean_) / self.std_
        elif self.method == 'normalize':
            return X / self.max_
        else:
            raise ValueError(f"Unknown method: {self.method}")


class PCAWithPreProcessing(BaseEstimator, TransformerMixin):
    """PCA with PreProcessing Pipeline."""

    def __init__(self, n_components=2, preprocessing_method='normalize', preprocessing_param=None):
        """Initialize the pipeline.

        Args:
        n_components (int): Number of principal components to retain.
        preprocessing_method (str): Pre-processing method (e.g., 'normalize', 'standardize').
        preprocessing_param (dict): Parameters for the pre-processing method.
        """
        self.n_components = n_components
        self.preprocessing_method = preprocessing_method
        self.preprocessing_param = preprocessing_param

    def fit(self, X, y=None):
        """Fit the pipeline to the data.

        Args:
        X (array-like): The input data to fit.
        y (array-like, optional): The target values (ignored).

        Returns:
        self: Fitted pipeline instance.
        """
        self.preprocessing_ = PreProcessingTransformer(
            method=self.preprocessing_method,
            param=self.preprocessing_param,
        )
        self.pca_ = PCA(n_components=self.n_components)

        X_preprocessed = self.preprocessing_.fit_transform(X)
        self.pca_.fit(X_preprocessed)
        return self

    def transform(self, X):
        """Apply the pipeline to transform the data.

        Args:
        X (array-like): The input data to transform.

        Returns:
        array-like: Transformed data.
        """
        X_preprocessed = self.preprocessing_.transform(X)
        return self.pca_.transform(X_preprocessed)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform the data using the pipeline.

        Args:
        X (array-like): The input data to fit and transform.
        y (array-like, optional): The target values (ignored).
        fit_params: Additional fitting parameters.

        Returns:
        array-like: Transformed data.
        """
        self.fit(X, y, **fit_params)
        return self.transform(X)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    pipeline = PCAWithPreProcessing(n_components=2, preprocessing_method='standardize')
    X_transformed = pipeline.fit_transform(X)
