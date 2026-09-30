"""
Script to create minimal test dataset structure for evaluate_render.py
Uses preprocessed data from /mnt/HDD1/bk_dataset
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Create base data directory
data_root = "./data"
test_dir = os.path.join(data_root, "test")

# Create directory structure
dirs_to_create = [
    os.path.join(test_dir, "Audio_files/RECOLA/group-1/P25"),
    os.path.join(test_dir, "Video_files/RECOLA/group-1/P25"),
    os.path.join(test_dir, "Emotion/RECOLA/group-1/P25"),
    os.path.join(test_dir, "3D_FV_files/RECOLA/group-1/P25"),
    os.path.join(test_dir, "Audio_files/RECOLA/group-1/P26"),
    os.path.join(test_dir, "Video_files/RECOLA/group-1/P26"),
    os.path.join(test_dir, "Emotion/RECOLA/group-1/P26"),
    os.path.join(test_dir, "3D_FV_files/RECOLA/group-1/P26"),
]

for dir_path in dirs_to_create:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created: {dir_path}")

# Create a simple test.csv with speaker-listener pairs
# Format: index, speaker_path, listener_path
csv_data = []
for i in range(1, 11):  # Create 10 samples for testing
    speaker_path = f"RECOLA/group-1/P25/{i}"
    listener_path = f"RECOLA/group-1/P26/{i}"
    csv_data.append([i, speaker_path, listener_path])

df = pd.DataFrame(csv_data, columns=['index', 'speaker', 'listener'])
csv_path = os.path.join(data_root, 'test.csv')
df.to_csv(csv_path, index=False)
print(f"\nCreated CSV file: {csv_path}")
print(f"Number of samples: {len(csv_data)}")

# Create neighbour_emotion_test.npy 
# This is used for finding appropriate reactions
n_samples = len(csv_data) * 2  # speaker + listener swapped
neighbour_emotion = np.eye(n_samples, dtype=bool)  # Identity matrix for simplicity
# Add some neighbors for diversity
for i in range(n_samples):
    if i + 1 < n_samples:
        neighbour_emotion[i, i+1] = True
    if i - 1 >= 0:
        neighbour_emotion[i, i-1] = True

neighbour_path = os.path.join(data_root, 'neighbour_emotion_test.npy')
np.save(neighbour_path, neighbour_emotion)
print(f"Created neighbour emotion file: {neighbour_path}")
print(f"Neighbour emotion shape: {neighbour_emotion.shape}")

# Create dummy files for testing
print("\nCreating dummy data files...")
for i in range(1, 11):
    # Audio files (78 dim MFCC features, 751 frames)
    audio_s = np.random.randn(751, 78).astype(np.float32)
    audio_l = np.random.randn(751, 78).astype(np.float32)
    
    # Emotion files (25 dim)
    emotion_s = np.random.randn(751, 25).astype(np.float32)
    emotion_l = np.random.randn(751, 25).astype(np.float32)
    
    # 3DMM files (58 dim) - Using actual dimensions from config
    dmm_s = np.random.randn(751, 58).astype(np.float32)
    dmm_l = np.random.randn(751, 58).astype(np.float32)
    
    # Save dummy audio (will be loaded via extract_audio_features, so create dummy wav)
    # For now, we'll skip actual wav files and modify the code if needed
    
    # Save emotion as .csv (as expected by dataset)
    pd.DataFrame(emotion_s).to_csv(
        os.path.join(test_dir, f"Emotion/RECOLA/group-1/P25/{i}.csv"),
        index=False, header=False
    )
    pd.DataFrame(emotion_l).to_csv(
        os.path.join(test_dir, f"Emotion/RECOLA/group-1/P26/{i}.csv"),
        index=False, header=False
    )
    
    # Save 3DMM as .npy
    np.save(
        os.path.join(test_dir, f"3D_FV_files/RECOLA/group-1/P25/{i}.npy"),
        dmm_s
    )
    np.save(
        os.path.join(test_dir, f"3D_FV_files/RECOLA/group-1/P26/{i}.npy"),
        dmm_l
    )
    
print(f"Created {i} dummy samples")

print("\n✅ Dataset structure created successfully!")
print("\nNext steps:")
print("1. You may need to disable audio/video loading in the config")
print("2. Run: python evaluate_render.py --epoch_num 500 --exp_num 1 --mode test --config configs/rewrite_weight.yaml")
