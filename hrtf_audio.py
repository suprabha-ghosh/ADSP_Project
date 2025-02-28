import pandas as pd
import numpy as np
import joblib
import soundfile as sf
import os
from scipy.signal import fftconvolve

def load_and_predict_hrtf(model_path, csv_path):
    """Load trained HRTF model and predict personalized HRTFs for participants."""
    try:
        model, scaler = joblib.load(model_path)
        print("✅ Model loaded successfully.")

        df = pd.read_csv(csv_path)
        df = df.set_index("Parameter").T.reset_index()
        df.rename(columns={"index": "Person"}, inplace=True)

        feature_cols = [col for col in df.columns if col != "Person"]
        X_new = df[feature_cols]

        X_scaled = scaler.transform(X_new)
        df["Predicted_HRTF"] = list(model.predict(X_scaled))

        df.to_csv("predicted_hrtf_values.csv", index=False)
        print("✅ Personalized HRTFs saved in 'predicted_hrtf_values.csv'.")
        
        return df

    except Exception as e:
        print(f"❌ Error in load_and_predict_hrtf: {str(e)}")
        raise

def hrtf_to_hrir(hrtf_magnitude):
    """Convert HRTF magnitude response to HRIR using IFFT."""
    hrir = np.fft.irfft(hrtf_magnitude)
    hrir = np.nan_to_num(hrir)  # Replace NaN/Inf with 0
    return hrir / (np.max(np.abs(hrir)) + 1e-9)  # Normalize

def generate_hrirs(df):
    """Convert predicted HRTFs to HRIRs for each participant."""
    try:
        df["HRIR_Left"] = df["Predicted_HRTF"].apply(hrtf_to_hrir)
        df["HRIR_Right"] = df["HRIR_Left"].apply(lambda x: np.roll(x, 10))  # ITD simulation
        return df
    except Exception as e:
        print(f"❌ Error in generate_hrirs: {str(e)}")
        raise

def apply_hrirs_to_audio(df, audio_file):
    """Apply HRIRs to mono audio and save one stereo binaural audio file per person."""
    try:
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"❌ Audio file not found: {audio_file}")

        audio, sr = sf.read(audio_file)
        if len(audio.shape) > 1:  # Convert to mono if stereo
            audio = np.mean(audio, axis=1)
        print(f"✅ Audio file loaded: {audio_file} | Sample rate: {sr} Hz.")

        output_dir = "Personalised_HRTF_audio"
        os.makedirs(output_dir, exist_ok=True)

        # Get unique person IDs (removing _Left and _Right suffixes)
        person_ids = sorted(list(set([p.split('_')[0] for p in df['Person']])))
        print(f"Found {len(person_ids)} unique participants")

        for person_id in person_ids:
            try:
                # Get left and right data for this person
                left_mask = df['Person'] == f"{person_id}_Left"
                right_mask = df['Person'] == f"{person_id}_Right"
                
                if not any(left_mask) or not any(right_mask):
                    print(f"⚠️ Missing data for person {person_id}")
                    continue

                left_row = df[left_mask].iloc[0]
                right_row = df[right_mask].iloc[0]

                # Get HRIRs for left and right ears
                left_hrir = left_row["HRIR_Left"]
                right_hrir = right_row["HRIR_Right"]

                # Process left and right channels
                left_ear = fftconvolve(audio, left_hrir, mode="same")
                right_ear = fftconvolve(audio, right_hrir, mode="same")

                # Normalize both channels together
                max_val = max(np.max(np.abs(left_ear)), np.max(np.abs(right_ear)))
                if max_val > 0:
                    left_ear /= max_val
                    right_ear /= max_val

                # Create stereo audio by combining left and right channels
                binaural_audio = np.column_stack((left_ear, right_ear))

                # Save one binaural (stereo) file per person
                output_file = os.path.join(output_dir, f"binaural_{person_id}.wav")
                sf.write(output_file, binaural_audio, sr)

                print(f"✅ Generated binaural audio for person {person_id}")

            except Exception as e:
                print(f"❌ Error processing person {person_id}: {str(e)}")
                continue

    except Exception as e:
        print(f"❌ Error in apply_hrirs_to_audio: {str(e)}")
        raise

def generate_binaural_audio(model_path, csv_path, audio_file):
    """Complete process: Load model, predict HRTFs, generate HRIRs, and apply to audio."""
    print("🔹 Step 1: Predicting Personalized HRTFs...")
    df = load_and_predict_hrtf(model_path, csv_path)

    print("🔹 Step 2: Converting HRTFs to HRIRs...")
    df = generate_hrirs(df)

    print("🔹 Step 3: Applying HRIRs to Audio...")
    apply_hrirs_to_audio(df, audio_file)

    print("✅ All binaural audio files generated successfully!")

if __name__ == "__main__":
    generate_binaural_audio("data/simple_hrtf_model.pkl", "data/ear_pair_parameters.csv", "data/original.wav")
