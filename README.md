# Cat-vs-Dog-Image-Classification-Using-Neural-Network
This project implements a neural network model using TensorFlow to predict whether an image contains a cat or a dog. The dataset used is a collection of labeled images of cats and dogs.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- TensorFlow
- Pandas
- Pillow
- Scikit-Learn
- Matplotlib
  
You can install the required libraries using the following commands:
```bash
pip install tensorflow pandas pillow scikit-learn matplotlib
```
## Usage

1. Clone the repository
```bash
git clone https://github.com/yourusername/Cat-vs-Dog-Image-Classification.git
cd Cat-vs-Dog-Image-Classification
```
2. Run the script:
```bash
python cat_dog_classification.py
```

## Project Structure
```bash
Cat-vs-Dog-Image-Classification/
├── cat_dog_classification.py       # Python script with the code
├── README.md                       # Project README file
└── requirements.txt                # List of dependencies
```

## Data preprocessing
The dataset consists of images stored in a directory. The filenames are used to extract labels (cat or dog). The images are resized to 150x150 pixels and normalized. The data is then split into training and validation sets.

```bash
from sklearn.model_selection import train_test_split

# Split the DataFrame into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
```

## Model Training
A convolutional neural network (CNN) is implemented using TensorFlow. The model consists of several convolutional and pooling layers, followed by fully connected layers. The model is trained using binary cross-entropy loss and the Adam optimizer.

```bash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Evaluation
The model's performance is evaluated on the validation set. Accuracy and loss are plotted for both training and validation data.
```bash
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Prediction
The trained model is used to predict the class of images in the test set. The predictions are saved to a CSV file.
```bash
predictions = model.predict(test_generator, steps=len(test_generator))
predicted_classes = (predictions > 0.5).astype(int).flatten()

test_data_df['prediction'] = predicted_classes
test_data_df.to_csv('test_predictions.csv', index=False)
```

## License
This project is licensed under the MIT License.

