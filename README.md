# Bharat_Intern_DataScience_project
Task 1: Stock Price Prediction with LSTM
This project centers around the utilization of Long Short-Term Memory (LSTM) neural networks to forecast stock prices. It makes use of historical stock price data from Tiingo for Nestlé (NSRGF). 
The following crucial steps are involved in this project:

1. Data Retrieval and Preprocessing:
   - Acquiring historical stock price data from Tiingo with an API key.
   - Loading the dataset into a Pandas DataFrame.
   - Gaining insights into the dataset structure using functions like info(), head(), and describe().
   - Checking for and identifying minimal missing data.

2. Data Visualization:
   - Extracting the 'date' and 'close' columns for further analysis.
   - Creating a time series plot to visualize the evolution of Nestlé's closing stock prices over time.
   - Configuring the x-axis to display years.

3. Data Preprocessing:
   - Normalizing the 'close' prices using Min-Max scaling to confine the data within the range of 0 to 1.
   - Splitting the dataset into training and testing sets, with 80% of the data reserved for training.

4. Model Construction:
   - Building a Sequential LSTM model for time series forecasting.
   - The model architecture includes two LSTM layers, each with 50 units, followed by two Dense layers with 25 and 1 unit(s) respectively.
   - Compiling the model using the 'adam' optimizer and Mean Squared Error (MSE) loss function.

5. Model Training:
   - Training the model on the training dataset with a batch size of 1 and one epoch.

6. Testing and Evaluation:
   - Preparing the test data by sliding a window of 60 historical data points.
   - Using the trained model to make predictions on the test data.
   - Inverse transforming predictions to obtain actual stock prices.
   - Calculating the Root Mean Squared Error (RMSE) to assess the model's accuracy in predicting stock prices.
   - Lower RMSE values indicate higher predictive accuracy.
   
This project serves as a demonstration of a workflow for time series forecasting using LSTM neural networks and offers valuable insights into Nestlé's stock price trends.

Task 2: Handwritten Digit Recognition with the MNIST Dataset

This project illustrates a Python script that employs a deep learning model to identify handwritten numbers. It makes use of the MNIST dataset for training and a pre-trained model to forecast the digits in images stored in a designated desktop folder. The code commences by loading and preparing the MNIST dataset, training a neural network model, and subsequently applying this model for digit prediction on images stored in the specified directory.

Import Essential Libraries:

- Utilize the 'os' library for file system operations.
- Employ 'cv2' for image processing.
- Utilize 'numpy' for numerical computations.
- Use 'matplotlib' for visualizing images.
- Rely on 'tensorflow' for machine learning and deep learning tasks.

Load and Preprocess the MNIST Dataset:

- Load the MNIST dataset, comprising images of handwritten digits paired with their labels.
- Normalize the pixel values of the images to fall within the 0 to 1 range, enhancing training efficiency.

Define a Neural Network Model:

- Create a Sequential model using TensorFlow/Keras.
- Flatten the 28x28 pixel images into a 1D array.
- Include two dense layers with ReLU activation functions for feature extraction.
- Append the output layer with 10 units and use softmax activation for digit classification.
- Compile the model using the Adam optimizer and sparse categorical cross-entropy loss.

Train the Model:

- Train the model utilizing the training data (x_train and y_train) over three epochs.
- The model acquires the ability to classify handwritten digits based on the training dataset.

Evaluate the Model:

- Assess the trained model's performance on the test data (x_test and y_test).
- Report the model's loss and accuracy on the test dataset.

Load and Predict on Desktop Images:

- Define the path to the folder containing digit images on the desktop.
- List all files in the folder with the ".png" extension.
- Iterate through the image files: Load each image using OpenCV and extract the first channel (assuming grayscale).
- Modify the image colors if necessary.
- Employ the trained model to predict the digit in the image.
- Display the image and the predicted digit using matplotlib.
- Handle any exceptions or errors that may arise during the process.

This code serves as a demonstration of the predictive capabilities of a neural network model trained on the MNIST dataset. It illustrates the steps involved in loading, preprocessing, and predicting digits from a designated folder using the pre-trained model.
