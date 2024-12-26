Zoodentification

This project aims to identify animals among following species: 'Elephant', 'Giraffe', 'Lion', 'Tiger', 'Bear', 'Red panda', 'Kangaroo', 'Panda', 'Crocodile', 'Penguin', 'Jaguar (Animal)', 'Rhinoceros', 'Hippopotamus', 'Monkey'.

How to run?
- Use python 3.8 and compatible versions of required libraries.
- You can use get_data.py to download images from OpenImages dataset or use openimages_zoo_animals15 folder directly.
- Run main.py to train the model.
- Try predict.py to see how trained model works on test-images folder

Model Architecture
- Transfer learning model leveraging MobileNetV2 as the base.
- Designed for feature extraction and classification. All weights are fine tuned during process.
- 
Architecture Details
- Input Shape: 128x128x3.
- Output Shape: 14 classes (softmax activation).

Architecture Sequence
- Base Model: MobileNetV2 (pre-trained on ImageNet).
- Global Average Pooling Layer.
- Dense Layer with 128 units (ReLU activation).
- Dropout Layer (0.5).
- Output Layer: Dense with softmax activation for 14 classes.
