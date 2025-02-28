import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class HRTFPredictor:
    def __init__(self, model_path, frequencies=np.linspace(20, 20000, 1024)):
        self.frequencies = frequencies
        self.model, self.scaler = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained HRTF model and scaler"""
        try:
            model, scaler = joblib.load(model_path)
            print("Model loaded successfully")
            return model, scaler
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_data(self, csv_file):
        """Load and preprocess ear parameter data"""
        try:
            # Load data
            df = pd.read_csv(csv_file)
            print("\nOriginal data shape:", df.shape)
            
            # Transpose and structure the data
            df = df.set_index("Parameter").T.reset_index()
            df.rename(columns={"index": "Person"}, inplace=True)
            
            # Verify required features
            required_features = [
                'aspect_ratio',
                *[f'hu_moment_{i}' for i in range(1, 8)],
                *[f'fourier_descriptor_{i}' for i in range(1, 6)]
            ]
            
            missing_features = [feat for feat in required_features 
                              if feat not in df.columns]
            
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select features and check for missing values
            feature_cols = [col for col in df.columns if col != "Person"]
            X_new = df[feature_cols]
            
            if X_new.isnull().any().any():
                print("Warning: Missing values detected. Filling with mean values.")
                X_new = X_new.fillna(X_new.mean())
            
            return df, X_new
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise

    def predict_hrtfs(self, X_new):
        """Generate HRTF predictions"""
        try:
            # Scale features
            X_scaled = self.scaler.transform(X_new)
            
            # Generate predictions
            predicted_HRTFs = self.model.predict(X_scaled)
            
            # Validate predictions
            self.validate_predictions(predicted_HRTFs)
            
            return predicted_HRTFs
            
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            raise

    def validate_predictions(self, predictions):
        """Validate HRTF predictions"""
        # Convert to dB for analysis
        predictions_db = 20 * np.log10(np.abs(predictions))
        
        print("\nValidating predictions:")
        print(f"Shape: {predictions.shape}")
        print(f"Value range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"dB range: [{predictions_db.min():.1f}, {predictions_db.max():.1f}] dB")
        
        # Check for anomalies
        if np.any(np.isnan(predictions)):
            print("Warning: NaN values detected in predictions")
        if np.any(np.isinf(predictions)):
            print("Warning: Infinite values detected in predictions")
        if predictions_db.max() > 20 or predictions_db.min() < -60:
            print("Warning: Predictions outside typical HRTF range")

    def plot_hrtf_responses(self, predictions, output_path):
        """Plot HRTF frequency responses"""
        num_samples = min(5, len(predictions))  # Plot up to 5 examples
        
        plt.figure(figsize=(15, 3*num_samples))
        
        for i in range(num_samples):
            plt.subplot(num_samples, 1, i+1)
            
            # Plot magnitude response
            plt.semilogx(self.frequencies, 
                        20 * np.log10(np.abs(predictions[i])))
            
            plt.grid(True)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.title(f'HRTF Response - Sample {i+1}')
            plt.ylim(-60, 20)  # Typical HRTF range
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"\nHRTF response plots saved to {output_path}")

def main():
    # Paths
    MODEL_PATH = "data/simple_hrtf_model.pkl"
    CSV_PATH = "csv/ear_pair_parameters.csv"
    OUTPUT_DIR = "data"
    
    # Initialize predictor
    predictor = HRTFPredictor(MODEL_PATH)
    
    # Load and preprocess data
    df, X_new = predictor.preprocess_data(CSV_PATH)
    
    # Generate predictions
    predicted_HRTFs = predictor.predict_hrtfs(X_new)
    
    # Add predictions to DataFrame
    df["Predicted_HRTF"] = list(predicted_HRTFs)
    
    # Save predictions
    output_file = f"csv/predicted_hrtf_values.csv"
    df.to_csv(output_file, index=False)
    print(f"\nPredicted HRTFs saved to '{output_file}'")
    
    # Generate visualization
    predictor.plot_hrtf_responses(predicted_HRTFs, 
                                f"{OUTPUT_DIR}/hrtf_responses.png")

if __name__ == "__main__":
    main()