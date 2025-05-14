MNIST Digit Classifier with Streamlit

This project is a from-scratch implementation of a neural network that classifies handwritten digits from the MNIST dataset. The model is trained using NumPy only (no TensorFlow or PyTorch), and deployed via an interactive Streamlit web app where users can draw digits and get real-time predictions.

🚀 Demo
Draw a digit (0–9) on the canvas and the app will predict what it is!
![MnistExamples](https://github.com/user-attachments/assets/25e0e72c-3ea3-4744-9166-3e487fdbe8a2)


🛠️ Features

-Neural Network implemented from scratch

-Multiple experiments with:

-Activation functions: sigmoid, tanh

-Loss functions: MSE, cross-entropy

-Network architectures

-Accuracy and evaluation metrics: precision, recall, F1, confusion matrix

-Streamlit frontend for:

-Drawing your own digit

-Predicting top 2 digits with confidence scores

-Visualizing preprocessed 28x28 input


🧠 Model Training (Model.py)

-Uses sklearn.datasets.fetch_openml to load MNIST

-Custom training loop with:

-Forward and backward pass

-Weight initialization and updates

-Mini-batch gradient descent

-Evaluation metrics include accuracy, confusion matrix, precision, recall, and F1 score

-Final weights are saved to .npy files and used in the Streamlit app

🎨 Streamlit App (app.py)

-Users can draw digits using streamlit-drawable-canvas

-Preprocessing includes:

-Inversion, normalization, Gaussian smoothing

-Resizing and centering the digit

-Predicts top 2 digits with probabilities using the trained model

-Fully interactive UI with options to clear canvas and view processed image

📦 Installation

git clone https://github.com/1amIbrahim/Image-Classification.git

cd Image-Classification

python -m venv venv

venv\Scripts\activate  # On Windows

source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt

▶️ Run the App

Make sure the .npy weight files are present (or run Model.py to train and save them), then launch the app:

streamlit run app.py

📁 Project Structure

Image-Classification/
│
├── Model.py             # Neural network logic and training
├── app.py               # Streamlit UI for digit drawing and prediction
├── weight_layer_*.npy   # Trained weights
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore

📚 Dependencies

-numpy

-matplotlib

-scikit-learn

-scikit-image

-scipy

-streamlit

-streamlit-drawable-canvas

📊 Sample Output (Training)

Epoch 10: Loss = 0.0175, Accuracy = 95.86%

Confusion Matrix:

[[...]]

Accuracy: 95.12%

Macro Precision: 0.9512

Macro Recall: 0.9510

Macro F1 Score: 0.9509

🧪 Experiments Included

✅ Different hidden layers: [256,128,64], [512,256], [64,10], etc.

✅ Comparison between sigmoid and tanh

✅ MSE vs Cross-Entropy Loss

✅ Centering and preprocessing drawn digits for better accuracy

📌 To-Do / Future Ideas

✅ Add UI for architecture selection

📈 Plot training loss in real-time

📦 Convert to a Docker app

🌐 Deploy on Streamlit Cloud or HuggingFace Spaces

🧑‍💻 Author

1amIbrahim

Project done as a deep learning learning exercise.










