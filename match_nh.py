import numpy as np
import pandas as pd

# Load participant ear data and transpose it
df_participants = pd.read_csv("data/ear_pair_parameters.csv", index_col=0).T
# Now each parameter will be a column and each participant will be a row

# Load NH HRIR frequency features
df_nh_features = pd.read_csv("csv/nh_hrir_features.csv")

def find_best_match(participant):
    """Find the closest NH file based on frequency response features."""
    # Using Hu moments and Fourier descriptors for matching
    hu1 = participant["hu_moment_1"]
    hu2 = participant["hu_moment_2"]
    fd1 = participant["fourier_descriptor_1"]
    fd2 = participant["fourier_descriptor_2"]

    # Compute Euclidean distance between participant and NH subjects
    df_nh_features["Distance"] = np.sqrt(
        (df_nh_features["Left_FFT_1"] - fd1) ** 2 +
        (df_nh_features["Left_FFT_2"] - fd2) ** 2 +
        (df_nh_features["Left_Skew"] - hu1) ** 2 +
        (df_nh_features["Left_Kurtosis"] - hu2) ** 2
    )

    best_match = df_nh_features.loc[df_nh_features["Distance"].idxmin()]
    return best_match["NH_File"]

# Print shape and first few rows of the transposed DataFrame to verify
print("Shape of participant data:", df_participants.shape)
print("\nFirst few rows of participant data:")
print(df_participants.head())
print("\nColumns in participant data:")
print(df_participants.columns.tolist())

# Apply matching to all participants
df_participants["Best_NH_HRTF"] = df_participants.apply(find_best_match, axis=1)

# Save matched NH files
df_participants.to_csv("csv/matched_hrtf_files.csv")

print("\n✅ Matched each participant to the closest NH HRTF and saved to 'matched_hrtf_files.csv'")

# Print summary of matches
print("\nSummary of matches:")
print(df_participants["Best_NH_HRTF"].value_counts())
