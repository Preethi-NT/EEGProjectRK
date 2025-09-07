import os
import shutil

# Paths
src_dir = "data/eeg_files"
dst_dir = "data/eeg_files_2"

# Make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

# Walk through source directory
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith("_2.edf"):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.move(src_path, dst_path)
            print(f"Moved: {file} → {dst_dir}")

