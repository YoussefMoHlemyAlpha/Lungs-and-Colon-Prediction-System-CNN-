# Lungs-and-Colon-Prediction-System-CNN
![image](https://github.com/user-attachments/assets/e0676d6e-2c8a-40a9-9442-6cac5653e69e)
## Lung and Colon Cancer Classification Using CNN
This project involves developing a convolutional neural network (CNN) to classify histopathological images of lung and colon tissue into different cancerous and non-cancerous categories. The dataset used in this project contains high-resolution images of tissue samples, which are classified into three categories for lung tissue and two for colon tissue.

## Steps Involved:
## Library Imports: Essential libraries such as TensorFlow, OpenCV, Matplotlib, NumPy, and PIL are imported for image processing, data manipulation, and model development.

### Downloading the Dataset: The dataset is downloaded from Kaggle, which contains lung and colon histopathological images. The data is stored in separate directories for each class:

Colon images: adenocarcinoma (cancerous) and benign (non-cancerous).
Lung images: adenocarcinoma, squamous cell carcinoma, and benign tissue.
### Data Preprocessing:

Directory Structure: The images are read from the directory, and a dictionary is created to map each class to a label.
Resizing Images: All images are resized to a fixed shape of 80x80 pixels using OpenCV.
Labeling: Labels are created for each class (colon_aca, colon_n, lung_aca, lung_n, lung_scc).
Normalization: Pixel values are scaled to the range [0, 1] by dividing by 255.
Data Splitting: The dataset is split into training and testing sets with 85% used for training and 15% for testing using train_test_split.

### CNN Model Architecture: Two separate CNN models are created for lung and colon image classification:

### Conv2D Layers: Three convolutional layers are used to extract features, followed by max-pooling layers to reduce spatial dimensions.
Fully Connected Layers: A dense layer of 128 units is added to learn non-linear combinations of features.
Softmax Output: The final layer uses the softmax activation function to output class probabilities.
Model Compilation & Training:

The models are compiled using the Adam optimizer and sparse categorical cross-entropy loss.
The training process is performed for 10 epochs with a validation split to monitor performance.
Model Evaluation: The models are evaluated on the test set to calculate accuracy. Predictions are made for the test images, and confusion matrices are generated to assess classification performance.

### Visualization:

Random images from the test set are plotted alongside their predicted and true labels to visualize the model’s performance.
Training and validation accuracy/loss curves are plotted to analyze the model's learning behavior.
Confusion Matrix: A confusion matrix is created using seaborn to illustrate how well the model distinguishes between different classes.

### Model Saving and Testing:

The trained models are saved as .keras files.
A functionality is added to upload individual images and predict whether they are cancerous or not using the trained model.
### Summary:
This project successfully implements a CNN-based image classification system to detect cancerous and benign tissues from histopathological images. By using separate models for lung and colon datasets, it leverages deep learning techniques to automate the detection of cancer types based on image analysis. The project also includes detailed visualizations of the model's performance through accuracy/loss plots and confusion matrices, providing valuable insights into how well the model generalizes to unseen data.







