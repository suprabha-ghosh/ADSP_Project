import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

class SimpleHRTFModel:
    def __init__(self):
        self.frequencies = np.linspace(20, 20000, 1024)  # Frequency range
        self.model = None
        self.scaler = StandardScaler()

    def gaussian_window(self, M, std):
        """Create a Gaussian window"""
        n = np.arange(M) - (M - 1.0) / 2.0
        sigma = std
        return np.exp(-0.5 * (n / sigma)**2)

    def load_data(self, csv_path):
        """Load and prepare data from CSV"""
        # Read the CSV file (parameters are in rows)
        df = pd.read_csv(csv_path)
        
        print("Original data:")
        print(df.head())
        
        # Transpose the data so each column becomes a feature
        df_transposed = df.transpose()
        df_transposed.columns = df_transposed.iloc[0]
        df_transposed = df_transposed.iloc[1:]  # Remove the first row
        
        print("\nTransposed data:")
        print(df_transposed.head())
        
        return df_transposed

    def generate_hrtf_response(self, parameters):
        """Generate synthetic HRTF response from parameters"""
        try:
            # Extract parameters
            aspect_ratio = float(parameters['aspect_ratio'])
            hu_moments = [float(parameters[f'hu_moment_{i}']) for i in range(1, 8)]
            fourier_desc = [float(parameters[f'fourier_descriptor_{i}']) for i in range(1, 6)]
            
            # Generate base response
            response = np.zeros_like(self.frequencies)
            
            # Add pinna effects
            pinna_response = (1 + aspect_ratio) * np.exp(-self.frequencies / 5000)
            
            # Add concha resonance
            concha_width = int(len(self.frequencies) * 0.05)  # Width of concha effect
            concha_response = signal.windows.gaussian(len(self.frequencies), concha_width)
            
            # Combine responses
            response = pinna_response + 2 * concha_response
            
            # Add effects from hu moments
            for i, moment in enumerate(hu_moments):
                response += 0.1 * moment * np.sin(2 * np.pi * self.frequencies * (i + 1) / 20000)
            
            # Normalize
            response = response / np.max(np.abs(response))
            
            return response
            
        except Exception as e:
            print(f"Error in HRTF generation: {str(e)}")
            raise

    def train_model(self, csv_path):
        """Train the HRTF model"""
        # Load and prepare data
        data = self.load_data(csv_path)
        
        # Generate HRTF responses for training
        X = data.values
        y = np.array([self.generate_hrtf_response(row) for _, row in data.iterrows()])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining HRTF model...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training R² score: {train_score:.3f}")
        print(f"Testing R² score: {test_score:.3f}")
        
        # Save model
        joblib.dump((self.model, self.scaler), 'data\simple_hrtf_model.pkl')
        print("Model saved as 'simple_hrtf_model.pkl'")
        
        return X_test_scaled, y_test

    def plot_results(self, X_test_scaled, y_test, num_examples=3):
        """Plot actual vs predicted HRTF responses"""
        # Get the actual number of examples available
        num_available = min(len(y_test), num_examples)
        
        plt.figure(figsize=(15, 5*num_available))
        
        # Get predictions
        y_pred = self.model.predict(X_test_scaled)
        
        for i in range(num_available):
            # Plot frequency response
            plt.subplot(num_available, 1, i+1)
            plt.semilogx(self.frequencies, 20 * np.log10(np.abs(y_test[i])), 
                        label='Actual', alpha=0.7)
            plt.semilogx(self.frequencies, 20 * np.log10(np.abs(y_pred[i])), 
                        label='Predicted', alpha=0.7)
            plt.grid(True)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.title(f'HRTF Response Comparison - Example {i+1}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('data\hrtf_comparisons.png')
        plt.close()

def main():
    # File path
    csv_path = "csv\ear_pair_parameters.csv"  # Update with your CSV path
    
    # Create and train model
    hrtf_model = SimpleHRTFModel()
    X_test_scaled, y_test = hrtf_model.train_model(csv_path)
    
    # Plot results
    hrtf_model.plot_results(X_test_scaled, y_test)

if __name__ == "__main__":
    main()