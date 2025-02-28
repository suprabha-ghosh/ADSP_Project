import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal
import os
from scipy.stats import pearsonr

class HRTFComparison:
    def __init__(self):
        self.results = []
        self.output_dir = "comparison_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_audio(self, file_path):
        """Load audio file and return signal"""
        try:
            audio, sr = sf.read(file_path)
            # Convert stereo to mono if necessary
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None, None

    def compute_metrics(self, audio1, audio2, label1, label2):
        """Compute comparison metrics between two audio signals"""
        try:
            # Ensure same length
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            # Correlation coefficient
            correlation, _ = pearsonr(audio1, audio2)
            
            # Mean Square Error
            mse = np.mean((audio1 - audio2) ** 2)
            
            # Signal-to-Noise Ratio
            signal_power = np.mean(audio2 ** 2)
            noise_power = np.mean((audio2 - audio1) ** 2)
            snr = 10 * np.log10(signal_power/noise_power) if noise_power > 0 else float('inf')
            
            # Spectral difference
            f_audio1 = np.abs(np.fft.fft(audio1))
            f_audio2 = np.abs(np.fft.fft(audio2))
            spectral_diff = np.mean(np.abs(f_audio1 - f_audio2))
            
            return {
                f'correlation_{label1}_vs_{label2}': correlation,
                f'mse_{label1}_vs_{label2}': mse,
                f'snr_{label1}_vs_{label2}': snr,
                f'spectral_diff_{label1}_vs_{label2}': spectral_diff
            }
            
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            return None

    def plot_comparison(self, original, personalized, benchmark, sr, person_id, save_path):
        """Generate comparison plots for all three audio signals"""
        try:
            plt.figure(figsize=(15, 12))
            
            # Time domain plot
            plt.subplot(3, 1, 1)
            time = np.arange(len(original)) / sr
            plt.plot(time, original, label='Original', alpha=0.7)
            plt.plot(time, personalized, label='Personalised HRTF', alpha=0.7)
            plt.plot(time, benchmark, label='Benchmark HRTF', alpha=0.7)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Time Domain Comparison - Person {person_id}')
            plt.legend()
            plt.grid(True)
            
            # Frequency domain plot
            plt.subplot(3, 1, 2)
            f_original = np.abs(np.fft.fft(original))
            f_personalized = np.abs(np.fft.fft(personalized))
            f_benchmark = np.abs(np.fft.fft(benchmark))
            freq = np.fft.fftfreq(len(original), 1/sr)
            
            # Plot only positive frequencies up to Nyquist frequency
            mask = freq >= 0
            plt.semilogx(freq[mask], 20 * np.log10(f_original[mask]), 
                        label='Original', alpha=0.7)
            plt.semilogx(freq[mask], 20 * np.log10(f_personalized[mask]), 
                        label='Personalised HRTF', alpha=0.7)
            plt.semilogx(freq[mask], 20 * np.log10(f_benchmark[mask]), 
                        label='Benchmark HRTF', alpha=0.7)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.title(f'Frequency Domain Comparison - Person {person_id}')
            plt.legend()
            plt.grid(True)
            
            # Spectrogram comparison
            plt.subplot(3, 1, 3)
            plt.specgram(personalized - benchmark, Fs=sr, cmap='coolwarm')
            plt.colorbar(label='Magnitude (dB)')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Difference Spectrogram (Personalised - Benchmark)')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"Error generating plots: {str(e)}")

    def compare_hrtf_audio(self, original_audio_path, personalized_dir, benchmark_dir):
        """Compare original, personalized and benchmark HRTF audio files"""
        try:
            # Load original audio
            original_audio, original_sr = self.load_audio(original_audio_path)
            if original_audio is None:
                raise ValueError("Could not load original audio file")

            # Get list of personalized audio files
            personalized_files = [f for f in os.listdir(personalized_dir) 
                                if f.startswith('binaural_') and f.endswith('.wav')]
            
            for p_file in personalized_files:
                # Extract person ID
                person_id = p_file.split('_')[1].split('.')[0]
                
                # Construct benchmark file path
                b_file = f"benchmark_hrtf_{person_id}.wav"
                b_path = os.path.join(benchmark_dir, b_file)
                
                if not os.path.exists(b_path):
                    print(f"Benchmark file not found for {person_id}")
                    continue
                
                # Load audio files
                p_audio, p_sr = self.load_audio(os.path.join(personalized_dir, p_file))
                b_audio, b_sr = self.load_audio(b_path)
                
                if p_audio is None or b_audio is None:
                    continue
                
                if not (p_sr == b_sr == original_sr):
                    print(f"Sample rate mismatch for person {person_id}")
                    continue
                
                # Compute metrics for all combinations
                metrics = {}
                comparisons = [
                    (original_audio, p_audio, 'original', 'personalized'),
                    (original_audio, b_audio, 'original', 'benchmark'),
                    (p_audio, b_audio, 'personalized', 'benchmark')
                ]
                
                for audio1, audio2, label1, label2 in comparisons:
                    comparison_metrics = self.compute_metrics(audio1, audio2, label1, label2)
                    if comparison_metrics:
                        metrics.update(comparison_metrics)
                
                if metrics:
                    metrics['person_id'] = person_id
                    self.results.append(metrics)
                
                # Generate plots
                plot_path = os.path.join(self.output_dir, f"comparison_person_{person_id}.png")
                self.plot_comparison(original_audio, p_audio, b_audio, p_sr, person_id, plot_path)
                
            # Create and save results DataFrame
            if self.results:
                df = pd.DataFrame(self.results)
                csv_path = os.path.join(self.output_dir, "hrtf_comparison_results.csv")
                df.to_csv(csv_path, index=False)
                print(f"\nResults saved to {csv_path}")
                
                # Generate summary plots
                self.plot_summary(df)
            else:
                print("No comparison results generated")
                
        except Exception as e:
            print(f"Error in comparison process: {str(e)}")

    def plot_summary(self, df):
        """Generate summary plots for all metrics"""
        try:
            # Get all metric columns
            metric_cols = [col for col in df.columns if col != 'person_id']
            
            # Calculate number of subplots needed
            n_metrics = len(metric_cols)
            n_rows = (n_metrics + 2) // 3  # 3 plots per row
            
            plt.figure(figsize=(15, 5*n_rows))
            
            for i, metric in enumerate(metric_cols, 1):
                plt.subplot(n_rows, 3, i)
                plt.bar(df['person_id'], df[metric])
                plt.title(metric)
                plt.xlabel('Person ID')
                plt.ylabel('Value')
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "summary_metrics.png"))
            plt.close()
            
            # Create correlation matrix plot
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[metric_cols].corr()
            plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar()
            plt.xticks(range(len(metric_cols)), metric_cols, rotation=45, ha='right')
            plt.yticks(range(len(metric_cols)), metric_cols)
            plt.title('Metric Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "metric_correlations.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error generating summary plots: {str(e)}")

def main():
    # Directory paths
    original_audio = "data/original.wav"  # Path to original audio file
    personalized_dir = "Personalised_HRTF_audio"
    benchmark_dir = "Benchmark_HRTF_audio"
    
    # Create comparison object
    comparator = HRTFComparison()
    
    # Run comparison
    print("Starting HRTF audio comparison...")
    comparator.compare_hrtf_audio(original_audio, personalized_dir, benchmark_dir)
    print("Comparison complete!")

if __name__ == "__main__":
    main()
