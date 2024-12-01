from openimages.download import download_images

# Specify the directory where images should be saved
data_dir = 'openimages_zoo_animals15'
exclusions_path = 'openimages_zoo_animals_exclusions15'

# List of animal classes you want to download
animal_classes = [
    'Elephant', 'Giraffe', 'Lion', 'Tiger',
    'Bear', 'Red panda', 'Kangaroo', 'Panda', 'Crocodile',
    'Penguin', 'Jaguar (Animal)', 'Rhinoceros', 'Hippopotamus', 'Monkey'
]

# Download the images for each specified class
for animal in animal_classes:
    print(f"Downloading images for {animal}...")
    download_images(
        dest_dir=data_dir,
        class_labels=[animal],
        exclusions_path=exclusions_path,
        limit=1000
    )