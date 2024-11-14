from openimages.download import download_images

# Specify the directory where images should be saved
data_dir = 'openimages_zoo_animals'
exclusions_path = 'openimages_zoo_animals_exclusions'

# List of animal classes you want to download
animal_classes = ['Elephant', 'Giraffe', 'Zebra', 'Lion', 'Tiger']

# Download the images for each specified class
for animal in animal_classes:
    print(f"Downloading images for {animal}...")
    download_images(
        dest_dir=data_dir,
        class_labels=[animal],
        exclusions_path=exclusions_path,
        limit=100  # You can adjust this number based on your needs
    )