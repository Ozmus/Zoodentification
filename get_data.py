import openimages

# Specify the directory where images should be saved
data_dir = 'openimages_zoo_animals'

# List of animal classes you want to download
animal_classes = ['Elephant', 'Giraffe', 'Zebra', 'Lion', 'Tiger']

# Download the images for each specified class
for animal in animal_classes:
    print(f"Downloading images for {animal}...")
    openimages.download_images(
        data_dir=data_dir,
        class_labels=[animal],
        max_images=100  # You can adjust this number based on your needs
    )
