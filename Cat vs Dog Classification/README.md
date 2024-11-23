Cat vs Dog Image Classification
This project implements a Convolutional Neural Network (CNN) to classify images of cats and dogs using Keras and TensorFlow. The model was trained on the PetImages dataset, which contains images of cats and dogs. The goal of this project is to classify input images as either a cat or a dog.

Project Overview
The project involves the following steps:

Data Preprocessing:

Images are loaded and preprocessed into a format suitable for training.
The dataset is divided into training and test sets.
Image augmentation is used to improve model generalization.
Model Architecture:

A simple CNN model is designed using Keras with layers such as Convolutional layers, Max Pooling layers, Flatten, and Dense layers.
The model is trained on the preprocessed dataset.
Model Training:

The model is trained using the training set, and its performance is validated using the test set.
The training process involves 10 epochs.
Prediction:

Once the model is trained, it can be used to predict if a given image is of a cat or a dog.
Model Save:

The trained model is saved in .h5 and .keras formats for later use.
Features
Image Preprocessing: Resize, normalize, and augment the input images.
Model: Convolutional Neural Network (CNN) with 3 convolutional layers and 2 fully connected layers.
Prediction: Predicts whether an image is of a dog or a cat.
Model Saving: Save and load the model using Keras' native formats (.h5 or .keras).
Installation
Ensure you have Python 3.x installed along with the following libraries:

pandas
numpy
matplotlib
PIL
keras
tensorflow
scikit-learn
You can install the required libraries via pip:

bash
Copy code
pip install pandas numpy matplotlib pillow tensorflow scikit-learn tqdm
How to Use
1. Clone the Repository
bash
Copy code
git clone https://github.com/Vinoth0208/Deeplearning.git
cd cat-vs-dog-classification
2. Dataset Setup
Download the PetImages dataset from the source (or use your own dataset with a similar structure), and place it in the PetImages directory within the project folder. The structure of the dataset should look like this:

markdown
Copy code
PetImages/
    ├── Dog/
    └── Cat/
3. Training the Model
The model training can be started by running the following command in your terminal or command prompt:

bash
Copy code
python train_model.py
The training process will run for 10 epochs. You can modify the number of epochs and other hyperparameters in the script.

4. Using the Trained Model for Prediction
To classify an image, simply provide the image path to the model using the code below:

python
Copy code
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('catvsdogclassification.keras')

# Load and preprocess the input image
image_path = "path_to_image.jpg"  # replace with your image path
img = load_img(image_path, target_size=(128, 128))
img = np.array(img) / 255.0  # Normalize the image
img = img.reshape(1, 128, 128, 3)  # Reshape for prediction

# Make prediction
pred = model.predict(img)

# Display the prediction result
label = 'Dog' if pred[0] > 0.5 else 'Cat'
print(f"Prediction: {label}")
5. Model Saving and Loading
The trained model will be saved as catvsdogclassification.keras. You can load it later for further inference or retraining.

python
Copy code
model.save('catvsdogclassification.keras')
Results
The model achieves a validation accuracy of around 81% after 10 epochs of training. You can use this model for real-time classification or further fine-tuning.

License
This project is licensed under the MIT License - see the LICENSE file for details.

