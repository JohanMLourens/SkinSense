from PIL import Image, ImageOps
import os
import shutil

# Function to resize an image
def resize_image(src_image, size=(600, 450), bg_color="white"):
    # Resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.LANCZOS)
    
    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
    
    return new_image

# Directories
training_folder_name = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/Trainning'
train_folder = 'C:/Users/S_CSIS-Postgrad/Desktop/AI Project/SkinCancerData/TestDataResized'

# Desired size for the resized images
size = (300, 300)

# Remove the output folder if it exists, then recreate it
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
os.makedirs(train_folder)

# Function to process images in a folder
def process_folder(input_folder, output_folder, size):
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                with Image.open(file_path) as image:
                    resized_image = resize_image(image, size)
                    save_folder = os.path.join(output_folder, os.path.relpath(root, input_folder))
                    os.makedirs(save_folder, exist_ok=True)
                    save_as = os.path.join(save_folder, file_name)
                    resized_image.save(save_as)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Process the images
print('Transforming images...')
process_folder(training_folder_name, train_folder, size)
print('Done.')