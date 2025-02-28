import pysofaconventions as sofa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
import os
import pandas as pd

def normalize_audio(audio_data):
    """Normalize audio data to range [-1, 1]"""
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        return audio_data / max_val
    return audio_data

def process_hrtf_convolution(audio, hrir_data, position=0):
    """Apply HRTF convolution for both ears"""
    left_hrir = hrir_data[position, 0, :]
    right_hrir = hrir_data[position, 1, :]

    # Apply FFT Convolution
    left_ear = fftconvolve(audio, left_hrir, mode="same")
    right_ear = fftconvolve(audio, right_hrir, mode="same")

    # Combine and normalize
    binaural = np.column_stack((left_ear, right_ear))
    return normalize_audio(binaural)

def main():
    # Define paths
    sofa_dir = "SOFA_ItE_hrtf B"
    original_audio_file = "data/original.wav"
    output_dir = "Benchmark_HRTF_audio"
    sofa_filename = "hrtf_M_hrtf B.sofa"  # Common SOFA file name

    # Verify input files exist
    if not os.path.exists(original_audio_file):
        raise FileNotFoundError(f"Original audio file not found: {original_audio_file}")
    
    if not os.path.exists("matched_hrtf_files.csv"):
        raise FileNotFoundError("matched_hrtf_files.csv not found!")

    # Load participant-to-NH matching file
    try:
        df_matched = pd.read_csv("matched_hrtf_files.csv", index_col=0)
        print(f"Loaded matching data for {len(df_matched)} participants")
        
        # Print the first few rows to verify data
        print("\nFirst few rows of matching data:")
        print(df_matched.head())
        
    except Exception as e:
        raise Exception(f"Error loading matched_hrtf_files.csv: {str(e)}")

    # Load original mono audio
    try:
        audio, sr = sf.read(original_audio_file)
        if len(audio.shape) > 1:
            print("Converting stereo to mono...")
            audio = np.mean(audio, axis=1)
        print(f"Loaded audio file: {sr}Hz, {len(audio)} samples")
    except Exception as e:
        raise Exception(f"Error loading audio file: {str(e)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each participant
    successful = 0
    
    # Get unique person IDs
    person_ids = sorted(list(set([idx.split('_')[0] for idx in df_matched.index if 'person' in idx])))
    
    for person in person_ids:
        try:
            # Get NH file for this person
            person_left = f"{person}_Left"
            if person_left not in df_matched.index:
                print(f"Warning: No data found for {person_left}")
                continue
                
            nh_folder = df_matched.loc[person_left, "Best_NH_HRTF"]  # This contains the NH folder name (e.g., NH71)
            
            # Construct SOFA file path
            sofa_file = os.path.join(sofa_dir, nh_folder, sofa_filename)
            
            # Debug print
            print(f"\nProcessing {person}")
            print(f"Looking for SOFA file: {sofa_file}")
            
            if not os.path.exists(sofa_file):
                print(f"Warning: SOFA file not found: {sofa_file}")
                continue

            # Load SOFA file
            hrtf = sofa.SOFAFile(sofa_file, "r")
            hrir_data = hrtf.getDataIR()
            sampling_rate = hrtf.getSamplingRate()

            # Verify sampling rates match
            if abs(sampling_rate - sr) > 1:
                print(f"Warning: Sample rate mismatch for {person}. HRTF: {sampling_rate}Hz, Audio: {sr}Hz")

            # Process audio
            binaural_audio = process_hrtf_convolution(audio, hrir_data)

            # Save output
            output_file = os.path.join(output_dir, f"benchmark_hrtf_{person}.wav")
            sf.write(output_file, binaural_audio, sr)
            
            successful += 1
            print(f"✅ Generated binaural audio for {person}")

        except Exception as e:
            print(f"❌ Error processing participant {person}: {str(e)}")
            continue

    print(f"\nProcessing complete: Successfully generated {successful}/{len(person_ids)} binaural files")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
