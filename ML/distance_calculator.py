import numpy as np

class DistanceCalculator:
    def __init__(self, num_features, num_components=3, max_history=10):
        self.num_features = num_features
        self.num_components = num_components
        self.max_history = max_history
        self.mu = None
        self.P_d = None
        self.recent_shifts = []

    def initialize_covariance_matrix(self, data):
        self.mu = np.mean(data, axis=0)
        centered_data = data - self.mu
        covariance_matrix = np.dot(centered_data.T, centered_data) / len(data)
        self.P_d = self.perform_eigendecomposition(covariance_matrix)

    def perform_eigendecomposition(self, covariance_matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1][:self.num_components]
        return eigenvectors[:, idx]

    def project_data(self, x_t):
        centered_x = x_t - self.mu
        return np.dot(self.P_d.T, centered_x)

    def calculate_shift_distance(self, y_t, y_t_minus_1):
        return np.linalg.norm(y_t - y_t_minus_1)

    def update_shifts(self, d_t):
        self.recent_shifts.append(d_t)
        if len(self.recent_shifts) > self.max_history:
            self.recent_shifts.pop(0)

    def calculate_relative_shift_magnitude(self):
        weights = np.ones(len(self.recent_shifts))  # Equal weighting for simplicity
        mu_d = np.average(self.recent_shifts, weights=weights)
        sigma_d = np.std(self.recent_shifts)
        return mu_d, sigma_d

    def classify_shift(self, d_t, alpha=1.96):
        mu_d, sigma_d = self.calculate_relative_shift_magnitude()
        M = (d_t - mu_d) / sigma_d
        if M < alpha:
            return 'Slight shift'
        else :
            return 'Severe shift'

