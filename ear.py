import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def capture_or_load_image(image_path=None):
    if image_path and os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        return img
    else:
        print(f"Image file not found: {image_path}")
        return None

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return gray, edges

def extract_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def normalize_hu_moments(hu_moments):
    """
    Normalize Hu moments using log transformation and round to 2 decimal places
    """
    return [round(-1 * np.copysign(1.0, h) * np.log10(abs(h)), 2) for h in hu_moments]

def extract_features(contours):
    if not contours:
        return None, None
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate aspect ratio and round to 2 decimals
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = round(float(h) / w, 2)
    
    hull = cv2.convexHull(largest_contour)
    
    # Calculate Hu moments
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_normalized = normalize_hu_moments(hu_moments)
    
    # Calculate Fourier descriptors and round to 2 decimals
    contour_complex = np.array(largest_contour[:, 0, 0] + 1j * largest_contour[:, 0, 1])
    fourier_descriptors = np.fft.fft(contour_complex)
    fd_normalized = np.abs(fourier_descriptors)[:10] / np.abs(fourier_descriptors)[0]
    fd_normalized = [round(fd, 2) for fd in fd_normalized]
    
    return {
        "aspect_ratio": aspect_ratio,
        "hu_moment_1": hu_moments_normalized[0],
        "hu_moment_2": hu_moments_normalized[1],
        "hu_moment_3": hu_moments_normalized[2],
        "hu_moment_4": hu_moments_normalized[3],
        "hu_moment_5": hu_moments_normalized[4],
        "hu_moment_6": hu_moments_normalized[5],
        "hu_moment_7": hu_moments_normalized[6],
        "fourier_descriptor_1": fd_normalized[0],
        "fourier_descriptor_2": fd_normalized[1],
        "fourier_descriptor_3": fd_normalized[2],
        "fourier_descriptor_4": fd_normalized[3],
        "fourier_descriptor_5": fd_normalized[4],
    }, hull, largest_contour


def display_results(img_left, contours_left, hull_left, edges_left,
                   img_right, contours_right, hull_right, edges_right, person_id):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    
    # Left ear original with contours
    output_left = img_left.copy()
    cv2.drawContours(output_left, contours_left, -1, (0, 255, 0), 2)
    cv2.drawContours(output_left, [hull_left], -1, (0, 0, 255), 2)
    ax1.imshow(cv2.cvtColor(output_left, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"{person_id} - Left Ear Contours")
    ax1.axis("off")
    
    # Left ear processed (edges)
    ax2.imshow(edges_left, cmap='gray')
    ax2.set_title(f"{person_id} - Left Ear Processed")
    ax2.axis("off")
    
    # Right ear original with contours
    output_right = img_right.copy()
    cv2.drawContours(output_right, contours_right, -1, (0, 255, 0), 2)
    cv2.drawContours(output_right, [hull_right], -1, (0, 0, 255), 2)
    ax3.imshow(cv2.cvtColor(output_right, cv2.COLOR_BGR2RGB))
    ax3.set_title(f"{person_id} - Right Ear Contours")
    ax3.axis("off")
    
    # Right ear processed (edges)
    ax4.imshow(edges_right, cmap='gray')
    ax4.set_title(f"{person_id} - Right Ear Processed")
    ax4.axis("off")
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "processed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{person_id}_processed.png"))
    plt.show()

def find_ear_pairs(base_directory):
    ear_pairs = {}
    person_dirs = [d for d in os.listdir(base_directory) 
                  if os.path.isdir(os.path.join(base_directory, d))]
    
    for person_dir in sorted(person_dirs):
        person_path = os.path.join(base_directory, person_dir)
        left_ear = None
        right_ear = None
        
        for file in os.listdir(person_path):
            file_lower = file.lower()
            full_path = os.path.join(person_path, file)
            
            if os.path.isfile(full_path):
                if 'left' in file_lower:
                    left_ear = full_path
                elif 'right' in file_lower:
                    right_ear = full_path
        
        if left_ear and right_ear:
            ear_pairs[person_dir] = (left_ear, right_ear)
        else:
            print(f"Warning: Missing ear image(s) for {person_dir}")
    
    return ear_pairs

def main(base_directory, output_csv_path):
    ear_pairs = find_ear_pairs(base_directory)
    
    if not ear_pairs:
        print("No ear pairs found!")
        return
    
    print(f"Found {len(ear_pairs)} persons with ear pairs")
    
    # Initialize dictionary to store all features
    all_features = {}
    
    for person_id, (left_path, right_path) in ear_pairs.items():
        print(f"\nProcessing {person_id}")
        
        # Process left ear
        img_left = capture_or_load_image(left_path)
        if img_left is None:
            continue
            
        # Process right ear
        img_right = capture_or_load_image(right_path)
        if img_right is None:
            continue
        
        # Extract features for both ears
        gray_left, edges_left = preprocess_image(img_left)
        gray_right, edges_right = preprocess_image(img_right)
        
        contours_left = extract_contours(edges_left)
        contours_right = extract_contours(edges_right)
        
        features_left, hull_left, largest_contour_left = extract_features(contours_left)
        features_right, hull_right, largest_contour_right = extract_features(contours_right)
        
        if features_left and features_right:
            # Store features for this person
            all_features[person_id] = {
                'left': features_left,
                'right': features_right
            }
            
            # Display and save processed images
            display_results(img_left, [largest_contour_left], hull_left, edges_left,
                          img_right, [largest_contour_right], hull_right, edges_right,
                          person_id)
        else:
            print(f"Failed to extract features for {person_id}")

    if all_features:
        # Create DataFrame with parameters as rows and ears as columns
        data = []
        parameters = list(next(iter(all_features.values()))['left'].keys())
        
        # Add parameters as rows
        for param in parameters:
            row = {'Parameter': param}
            for person_id in all_features:
                row[f'{person_id}_Left'] = all_features[person_id]['left'][param]
                row[f'{person_id}_Right'] = all_features[person_id]['right'][param]
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to CSV
        df.to_csv(output_csv_path, index=False)
        print(f"\nData saved to '{output_csv_path}'")
        print("\nExtracted Parameters:")
        print(df)
    else:
        print("No data was extracted from the images.")

if __name__ == "__main__":
    BASE_DIRECTORY = "ear_database"  # Your base directory containing person folders
    OUTPUT_CSV_PATH = "csv/ear_pair_parameters.csv"
    
    main(BASE_DIRECTORY, OUTPUT_CSV_PATH)
