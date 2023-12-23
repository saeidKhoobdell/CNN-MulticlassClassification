
Image Classification with PyTorch.

This project is focused on building an image classification model using PyTorch. The goal is to classify images of dogs into three classes: affenpinscher, akita, and corgi.

Project Structure
The project consists of the following components:

Data Loading and Preprocessing:

The training and testing datasets are loaded using PyTorch's ImageFolder class.
Image transformations, including resizing, random rotation, conversion to grayscale, normalization, and conversion to PyTorch tensors, are applied to the images.
Neural Network Model:

The neural network model, ImageMulticlassClassificationNet, is implemented using PyTorch's nn.Module class.
The model architecture includes two convolutional layers, max-pooling layers, and fully connected layers with ReLU activation functions.
The final layer uses the LogSoftmax activation function for multiclass classification.
Loss Function and Optimizer:

Cross-entropy loss is used as the loss function for training the classification model.
The Adam optimizer is employed to optimize the model's parameters.
Training:

The model is trained for a specified number of epochs.
The training loop iterates through batches of data, computes the loss, performs backpropagation, and updates the model's parameters.
Testing:

The trained model is evaluated on a separate test set.
Accuracy is calculated, and a confusion matrix is generated to assess the model's performance.
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/saeidKhoobdell/CNN-MulticlassClassification.git

Install the required packages:

bash
Copy code
pip install torch torchvision matplotlib numpy scikit-learn
Run the Python script:

bash
Copy code
python MulticlassClassification.py


Results
After training and testing the model, the accuracy and confusion matrix results are displayed. These metrics provide insights into the model's performance on the given classification task.

Feel free to experiment with hyperparameters, model architecture, or other aspects to further improve the model's accuracy.

Author
Saeid Khoobdel
License
This project is licensed under the MIT License.
