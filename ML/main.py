import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
from data_reader import get_data_loader
from incremental_model import MLPModel as IncrementalMLPModel, train as train_incremental, predict as predict_incremental
from batch_model import Batch_MLPModel, train_batch_model, predict as predict_batch_model
from distance_calculator import DistanceCalculator
from adaptive_window import AdaptiveWindow

def gaussian_kernel(distance, sigma=1.0):
    return np.exp(-distance**2 / (2 * sigma**2))

class ModelHistory:
    def __init__(self):
        self.history = {}

    def add_model(self, model_state, data_features):
        self.history[tuple(data_features.flatten())] = model_state

    def find_closest_model(self, current_features):
        min_distance = float('inf')
        closest_state = None
        for features, state in self.history.items():
            distance = np.linalg.norm(np.array(features) - current_features.flatten())
            if distance < min_distance:
                min_distance = distance
                closest_state = state
        return closest_state

def main():
    parser = argparse.ArgumentParser(description='Run the machine learning model on specified dataset.')
    parser.add_argument('filepath', type=str, help='Path to your data file.')
    args = parser.parse_args()

    batch_size = 1024
    num_features = 10
    num_classes = 2

    data_loader = get_data_loader(args.filepath, batch_size)
    incremental_model = IncrementalMLPModel(num_features, num_classes)
    batch_model = Batch_MLPModel(num_features, num_classes)
    dist_calc = DistanceCalculator(num_features)
    model_history = ModelHistory()
    adaptive_window = AdaptiveWindow(window_size=5, max_batches=5)

    last_batch_data = None
    accuracies = []

    for batch_data in data_loader:
        features, labels = batch_data[:, :-1], batch_data[:, -1]

        adaptive_window.add_batch(features.numpy())

        if last_batch_data is not None:
            shift_distance = dist_calc.calculate_shift_distance(features.numpy(), last_batch_data.numpy())
            shift_type = dist_calc.classify_shift(shift_distance)

            if shift_type == "Severe Shift":
                closest_state = model_history.find_closest_model(features.numpy())
                if closest_state:
                    # Load the closest model for prediction
                    retrieval_model = Batch_MLPModel(num_features, num_classes)
                    retrieval_model.load_state_dict(closest_state)
                    predicted_labels = predict_batch_model(retrieval_model, features)
                    accuracy = (predicted_labels == labels).float().mean()
                    accuracies.append(accuracy)
                    print("Retrieval Shift handled.")
                else:
                    # Handle regular Severe Shift using KMeans
                    kmeans = KMeans(n_clusters=num_classes).fit(features.numpy())
                    predicted_labels = kmeans.labels_
                    accuracy = (predicted_labels == labels.numpy()).mean()
                    accuracies.append(accuracy)
                    print("Severe Shift handled without retrieval.")

            elif shift_type == "Slight Shift":
                # Ensemble predictions from both models
                pred_inc = predict_incremental(incremental_model, features)
                pred_batch = predict_batch_model(batch_model, features)
                weights_inc = gaussian_kernel(shift_distance)
                weights_batch = gaussian_kernel(shift_distance)
                ensemble_pred = (weights_inc * pred_inc + weights_batch * pred_batch) / (weights_inc + weights_batch)
                accuracy = (ensemble_pred == labels).float().mean()
                accuracies.append(accuracy)
                print("Slight Shift handled.")

        # Always train the incremental model
        train_incremental(incremental_model, features, labels)

        # Update the batch model conditionally
        if adaptive_window.should_update():
            train_batch_model(batch_model, features, labels)
            model_history.add_model(batch_model.state_dict(), features.numpy())

        last_batch_data = features

    # Save accuracies to a file
    np.savetxt("accuracies.txt", np.array(accuracies))

if __name__ == "__main__":
    main()
