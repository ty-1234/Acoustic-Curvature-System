from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

class NeuralNetworkTester:
    def __init__(self, model_path, data_path):
        self.model = load_model(model_path)
        self.data = pd.read_csv(data_path)

    def preprocess_data(self):
        """
        Prepares the dataset for normal force predictions.
        """
        X = self.data.filter(like='normal_force').values  # Features for normal force
        y = self.data['force_normal'].values  # Target normal force
        return X, y

    def validate_model(self):
        """
        Validates the neural network model using Mean Absolute Error (MAE).
        """
        X, y = self.preprocess_data()
        predictions = self.model.predict(X).flatten()
        errors = y - predictions
        mae = np.mean(np.abs(errors))
        print(f"Validation Mean Absolute Error (MAE): {mae:.2f}")
        return mae

if __name__ == "__main__":
    tester = NeuralNetworkTester("tactile_sensor_model.h5", "collected_data.csv")
    tester.validate_model()
