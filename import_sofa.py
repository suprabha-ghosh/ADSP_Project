import pysofaconventions as sofa
import numpy as np
import os
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# Path to ARI HRTF dataset
sofa_dir = "SOFA_ItE_hrtf B"

def extract_hrir_features(sofa_file):
    """Extract spectral features from NH HRIR."""
    try:
        print(f"Processing file: {sofa_file}")
        hrtf = sofa.SOFAFile(sofa_file, "r")

        # Extract HRIR Data
        hrir_data = hrtf.getDataIR()
        print(f"HRIR data shape: {hrir_data.shape}")

        # Compute frequency-domain features from HRIR
        left_hrir = hrir_data[0, 0, :]  # First position, Left ear
        right_hrir = hrir_data[0, 1, :]  # First position, Right ear

        # Compute FFT and extract features
        left_fft = np.abs(fft(left_hrir))[:50]
        right_fft = np.abs(fft(right_hrir))[:50]

        # Compute statistical descriptors
        left_skew = skew(left_fft)
        right_skew = skew(right_fft)

        left_kurtosis = kurtosis(left_fft)
        right_kurtosis = kurtosis(right_fft)

        result = {
            "NH_File": os.path.basename(os.path.dirname(sofa_file)),  # Gets the subfolder name (NH2, NH3, etc.)
            "Left_FFT_1": left_fft[0],
            "Left_FFT_2": left_fft[1],
            "Left_Skew": left_skew,
            "Left_Kurtosis": left_kurtosis,
            "Right_FFT_1": right_fft[0],
            "Right_FFT_2": right_fft[1],
            "Right_Skew": right_skew,
            "Right_Kurtosis": right_kurtosis,
        }
        print(f"Extracted features for {result['NH_File']}")
        return result

    except Exception as e:
        print(f"Error processing {sofa_file}: {e}")
        return None

# Find all SOFA files in subfolders
sofa_files = []
for root, dirs, files in os.walk(sofa_dir):
    for file in files:
        if file.endswith('.sofa'):
            sofa_files.append(os.path.join(root, file))

print(f"Found {len(sofa_files)} SOFA files")

# Process all NH SOFA files
metadata_list = []
for file in sofa_files:
    features = extract_hrir_features(file)
    if features is not None:
        metadata_list.append(features)

print(f"Processed {len(metadata_list)} files successfully")

# Save extracted HRIR features to CSV
if metadata_list:
    df_nh_features = pd.DataFrame(metadata_list)
    print(f"DataFrame shape: {df_nh_features.shape}")
    df_nh_features.to_csv("csv/nh_hrir_features.csv", index=False)
    print("✅ Extracted HRIR spectral features from NH subjects and saved to 'nh_hrir_features.csv'")
else:
    print("❌ No data was processed successfully")
