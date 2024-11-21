import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

class GaussianMethods:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = pd.read_csv(data_file)
        self.X, self.y = self.preprocess_data()

    def preprocess_data(self):
        """
        Extracts normal force data for analysis.
        """
        X = self.data.filter(like='normal_force').values  # Features
        y = self.data['force_normal'].values  # Target
        return X, y

    def smooth_signal(self, signal, sigma=1.0):
        """
        Applies Gaussian smoothing to a signal.
        """
        return gaussian_filter1d(signal, sigma=sigma)

    def classify_with_gmm(self, n_components=3):
        """
        Classifies the dataset using a Gaussian Mixture Model (GMM).
        """
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(self.X)
        labels = gmm.predict(self.X)
        return labels

    def gaussian_basis_regression(self, test_size=0.2, gamma=1.0, alpha=1.0):
        """
        Performs regression using Gaussian basis functions.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        rbf_sampler = RBFSampler(gamma=gamma, n_components=100, random_state=42)
        X_train_transformed = rbf_sampler.fit_transform(X_train)
        X_test_transformed = rbf_sampler.transform(X_test)

        model = Ridge(alpha=alpha)
        model.fit(X_train_transformed, y_train)
        predictions = model.predict(X_test_transformed)
        mae = mean_absolute_error(y_test, predictions)
        print(f"Gaussian Basis Regression MAE: {mae:.2f}")
        return model, mae

if __name__ == "__main__":
    gm = GaussianMethods("collected_data.csv")
    labels = gm.classify_with_gmm(n_components=3)
    print("GMM Labels:", labels)
    model, mae = gm.gaussian_basis_regression()
    print(f"Trained regression model with MAE: {mae:.2f}")
