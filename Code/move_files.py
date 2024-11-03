import os
import shutil


data_dir = '/home/ubuntu/nlp_project/Data/'
images_dir = os.path.join(data_dir, 'images')


if not os.path.exists(images_dir):
    os.makedirs(images_dir)


image_extensions = ('.jpg', '.jpeg', '.png')

for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)


    if os.path.isfile(file_path) and file.lower().endswith(image_extensions):
        shutil.move(file_path, images_dir)
        print(f"Moved {file} to {images_dir}")


