import os
import zipfile
import shutil

zip_file_path = r"C:\Users\HP\Desktop\eeg-during-mental-arithmetic-tasks-1.0.0.zip"
extract_temp = r"data\temp_extracted"
final_eeg_path = r"data\eeg_files"
final_csv_path = r"data\subject-info.csv"

# Create temp extraction folder
os.makedirs(extract_temp, exist_ok=True)
os.makedirs(final_eeg_path, exist_ok=True)
os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)

# Extract zip
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_temp)

# Move files
for root, dirs, files in os.walk(extract_temp):
    for file in files:
        file_path = os.path.join(root, file)
        if file.endswith(".edf"):
            shutil.move(file_path, os.path.join(final_eeg_path, file))
        elif file.endswith(".csv"):
            shutil.move(file_path, final_csv_path)






