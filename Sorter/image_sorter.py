# Python script to sort through HAM10000 dataset and sort images into dedicated folders according to diagnosis

import pandas as pd
import os
import shutil

# Define the paths
excel_file = r'C:\Users\S_CSIS-PostGrad\Documents\Kaggle Skin Dataset\HAM10000_metadata.xlsx'  # Path to the Excel file
source_folder = r'C:\Users\S_CSIS-PostGrad\Documents\Kaggle Skin Dataset\HAM10000'  # Folder where the images are currently located
destination_base_folder = r'C:\Users\S_CSIS-PostGrad\Documents\Kaggle Skin Dataset\Sorted'  # Base folder where images will be moved

# Read the Excel file
df = pd.read_excel(excel_file)

# Assuming the Excel file has columns 'image_id' for image file name and 'dx' for diagnosed type
for index, row in df.iterrows():
    image_name = row['image_id']
    classification = row['dx']

    # Define the source and destination paths
    source_path = os.path.join(source_folder, image_name+ ".jpg")
    destination_folder = os.path.join(destination_base_folder, classification)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Define the destination path
    destination_path = os.path.join(destination_folder, image_name+ ".jpg")

    # Move the file
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved {image_name} to {destination_folder}")
    else:
        print(f"File {image_name} not found in source folder.")
