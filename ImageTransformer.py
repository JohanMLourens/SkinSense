import os
import shutil
from PIL import Image, ImageOps

# Define the path to the original training folder
original_folder = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/ResizedTrainning'
# Define the path to the folder where augmented images will be saved
augmented_folder = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/ResizedTrainningAugmented'

# Ensure the augmented folder exists
if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

# Iterate through each class folder in the original training folder
for class_folder in os.listdir(original_folder):
    class_folder_path = os.path.join(original_folder, class_folder)
    if os.path.isdir(class_folder_path):
        # Create the corresponding class folder in the augmented folder
        augmented_class_folder = os.path.join(augmented_folder, class_folder)
        if not os.path.exists(augmented_class_folder):
            os.makedirs(augmented_class_folder)

        # Iterate through each image in the class folder
        for image_name in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_name)
            if os.path.isfile(image_path):
                # Load the image
                image = Image.open(image_path)

                # Perform horizontal flip and save
                flipped_horizontal = ImageOps.mirror(image)
                flipped_horizontal_path = os.path.join(augmented_class_folder, f'h_{image_name}')
                flipped_horizontal.save(flipped_horizontal_path)

                # Perform vertical flip and save
                flipped_vertical = ImageOps.flip(image)
                flipped_vertical_path = os.path.join(augmented_class_folder, f'v_{image_name}')
                flipped_vertical.save(flipped_vertical_path)

                # Optionally, save the original image in the augmented folder as well
                original_copy_path = os.path.join(augmented_class_folder, image_name)
                shutil.copy(image_path, original_copy_path)

print("Augmentation completed. Flipped images saved to", augmented_folder)
