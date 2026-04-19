# Image Classification Project

This project compares two image classification approaches:

1. Traditional classification with feature extraction (`classification_traditionnelle.ipynb`)
2. Training a custom Convolutional Neural Network (CNN) (`cnn.ipynb`)
3. Feature extraction using pretrained models (`featExtractSingle2.py`)

## Project Structure

- `classification_traditionnelle.ipynb` -> SVM, k-NN, Naive Bayes, Decision Tree models
- `cnn.ipynb` -> Complete CNN model with TensorFlow/Keras
- `featExtractSingle2.py` -> Feature extraction script using VGG16, ResNet50, InceptionV3
- `datasets/` -> Folders containing images organized by class
- `features_csv/` -> CSV files with extracted feature vectors
- `saved_cnn_models/` -> Directory to store trained models
- `saved_classic_models/`
- `requirements.txt` -> File listing dependencies
- `README.md` -> Project overview (this file)

## Installation and Environment

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Upgrade pip:

```bash
pip install --upgrade pip
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Files and Features

### 1. classification_traditionnelle.ipynb

- Loads extracted features (.csv)
- Trains classical models (SVM, kNN, etc.)
- Displays metrics (accuracy, F1, precision, recall, AUC)
- Produces confusion matrices

### 2. cnn.ipynb

- Loads images using `image_dataset_from_directory`
- Trains a custom CNN (3 to 5 convolutional layers)
- Uses BatchNormalization, Dropout, Pooling, dense layers
- Displays training curves and metrics
- Allows model saving

### 3. featExtractSingle2.py

- Uses a pretrained model (VGG16, ResNet50, or InceptionV3)
- Extracts feature vectors from images
- Saves results into a CSV file
- Example command:

```bash
python featExtractSingle2.py --dataset_path ./datasets/Wildfire --model resnet50
```

## Expected Dataset Structure

Images must be organized as follows:

```text
datasets/
    DatasetName/
        class1/
        class2/
        ...
```

Each class contains images in JPG or PNG format.

## Usage Recommendations

- Pipeline 1 (traditional classification): recommended for simple datasets, few classes, or limited data.
- Pipeline 2 (custom CNN): recommended for more complex datasets with more classes and images.

## Author

Project carried out as part of the course INF5082 - Machine Learning and Data Mining.

Author: El Kolli Fares
