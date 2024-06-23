# Image Classification Using Decision Trees and CNNs

## Project Overview

This project investigates the application of machine learning techniques, specifically Decision Trees and Convolutional Neural Networks (CNNs), for classifying images in the Places365 dataset. The study explores both supervised and semi-supervised learning approaches to enhance classification accuracy.

## Requirements

To run the Python code, you need the following libraries:

- Python 3.6+
- scikit-learn
- numpy
- pandas
- matplotlib
- torch
- torchvision
- joblib

You can install the necessary libraries using:

```bash
pip install scikit-learn numpy pandas matplotlib torch torchvision joblib


# Instructions to Train/Validate the Model

## Training the Decision Tree Classifier

- Open and run `main.ipynb` to train the decision tree classifier on the Places365 dataset using both supervised and semi-supervised learning approaches.

## Training the CNN Model

- Open and run `cnn.ipynb` to train the CNN model on the Places365 dataset. The notebook includes data augmentation and early stopping techniques to prevent overfitting.

# Instructions to Run the Pre-trained Model

## Decision Tree Classifier

- Load the pre-trained decision tree model using Joblib:
  ```python
  import joblib
  model = joblib.load('path_to_saved_dt_model.joblib')


## Evaluate the model on the sample test dataset:

```python
from sklearn.metrics import classification_report

# Assuming X_test and y_test are already defined
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))


## CNN Model
### Load the pre-trained CNN model:

```python
import torch
model = torch.load('path_to_saved_cnn_model.pth')
model.eval()


## Evaluate the model on the sample test dataset:

```python
from torchvision import transforms
from torch.utils.data import DataLoader

# Define test transformations
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assuming test_dataset is already defined
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Compute and print classification report
from sklearn.metrics import classification_report
print(classification_report(all_labels, all_predictions))

## Source Code
The source code for both the Decision Tree and CNN models can be found in the following notebooks:

- `main.ipynb`: Training and evaluation of the Decision Tree Classifier.
- `adjusted_semi.ipynb`: Semi-supervised learning with Decision Trees.
- `cnn.ipynb`: Training and evaluation of the CNN model.

## Dataset
The Places365 dataset can be downloaded from [here](http://places2.csail.mit.edu/download.html). Follow the instructions on the page to obtain the dataset. Make sure to download the Places365_small dataset for this project.


Author(s): 
Supervised: Raghav Senwal 
CNN: Raghav Senwal
Semi-supervised: Alireza Lorestani, Nasim Fani

Trained models have been included in the src file, one can easily load them using joblib library and use the model to verify classification on unseen data.

Code example:
```
model_path = '<insert_external_path>/<insert_model_name>.joblib'
loaded_clf = joblib.load(model_path)
loaded_y_pred = loaded_clf.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)
print(f'Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%')
```
Trained model for CNN is uploaded on Moodle.

Code example to load model:
```
model.load_state_dict(torch.load('<model_name>.pth'))

```
