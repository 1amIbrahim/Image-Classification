import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data']
y = mnist['target']
y = y.astype(int)

def display_images(X, y):
    for i in range(5):
        idx = np.random.randint(0, X.shape[0])
        plt.imshow(X[idx].reshape(28,28), cmap='gray')
        plt.title(f"Label: {y[idx]}")
        plt.show()

def normalize(X):
    return X / 255.0

def divide(X, y):
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, X_test, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def mse(pred, target):
    return np.mean((pred - target)**2)

def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-8)) / pred.shape[0]

def one_hot_encode(y, num_classes=10):
    y = np.array(y)
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def initialize_weights(input_size, hidden_sizes, output_size):
    w = []
    w.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
    for i in range(len(hidden_sizes) - 1):
        w.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * 0.01)
    w.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
    return w

def forward_pass(x, weights, activation_fn):
    activations = []
    pre_activations = []
    a = x
    for i in range(len(weights)-1):
        z = a @ weights[i]
        pre_activations.append(z)
        a = activation_fn(z)
        activations.append(a)
    z = a @ weights[-1]
    pre_activations.append(z)
    a = softmax(z)
    activations.append(a)
    return activations, pre_activations

def backward_pass(x, y, activations, pre_activations, weights, activation_fn_derivative):
    grads = []
    y_true = one_hot_encode(y)
    delta = activations[-1] - y_true
    grad = activations[-2].T @ delta
    grads.insert(0, grad)
    for i in range(len(weights)-2, 0, -1):
        delta = (delta @ weights[i+1].T) * activation_fn_derivative(pre_activations[i])
        grad = activations[i-1].T @ delta
        grads.insert(0, grad)
    delta = (delta @ weights[1].T) * activation_fn_derivative(pre_activations[0])
    grad = x.T @ delta
    grads.insert(0, grad)
    return grads

def update_weights(weights, grads, lr):
    for i in range(len(weights)):
        weights[i] = weights[i] - lr * grads[i]
    return weights

def predict(x, weights, activation_fn):
    activations, _ = forward_pass(x, weights, activation_fn)
    return np.argmax(activations[-1], axis=1)

def compute_loss(predictions, labels, loss_fn):
    one_hot = one_hot_encode(labels)
    return loss_fn(predictions, one_hot)

def compute_accuracy(pred_labels, true_labels):
    return np.mean(pred_labels == true_labels)

def create_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_batches = np.array_split(X, X.shape[0] // batch_size)
    y_batches = np.array_split(y, y.shape[0] // batch_size)
    return X_batches, y_batches

def train(X_train, y_train, weights, activation_fn, activation_fn_derivative, loss_fn, epochs=10, batch_size=64, lr=0.01):
    for epoch in range(epochs):
        X_batches, y_batches = create_batches(X_train, y_train, batch_size)
        for x_batch, y_batch in zip(X_batches, y_batches):
            activations, pre_activations = forward_pass(x_batch, weights, activation_fn)
            grads = backward_pass(x_batch, y_batch, activations, pre_activations, weights, activation_fn_derivative)
            weights = update_weights(weights, grads, lr)
        preds = predict(X_train, weights, activation_fn)
        activations, _ = forward_pass(X_train, weights, activation_fn)
        loss = compute_loss(activations[-1], y_train, loss_fn)
        acc = compute_accuracy(preds, y_train)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc*100:.2f}%")
    return weights

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate_model(X_test, y_test, weights, activation_fn):
    preds = predict(X_test, weights, activation_fn)
    cm = confusion_matrix(y_test, preds)
    acc = compute_accuracy(preds, y_test)
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")

    return cm, acc, precision, recall, f1


if __name__ == "__main__":

    X = normalize(X)
    X_train, X_test, y_train, y_test = divide(X, y)

    print("Task12-A")
    weights1 = initialize_weights(784, [256, 128, 64], 10)
    trained_weights1 = train(X_train, y_train, weights1, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.01)
    for i, w in enumerate(trained_weights1):
        np.save(f"weight_layer_{i}.npy", w)


    evaluate_model(X_test,y_test,trained_weights1,sigmoid)
    print("Task12 - A")
    weights1 = initialize_weights(784, [256, 128, 64], 10)
    trained_weights1 = train(X_train, y_train, weights1, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.001)
    #Task12-B
    print("Task12 - B")
    weights2 = initialize_weights(784, [512, 256], 10)
    trained_weights2 = train(X_train, y_train, weights2, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.01) 

    print("Task12 - C")
    weights3 = initialize_weights(784, [256, 128, 64], 10)
    trained_weights3 = train(X_train, y_train, weights3, tanh, tanh_derivative, mse, epochs=10, batch_size=64, lr=0.01)
    for i, w in enumerate(trained_weights3):
        np.save(f"weight_layer_{i}.npy", w)


    #Task12-D
    print("Task12 - D")
    weights4 = initialize_weights(784, [256, 128, 64], 10)
    trained_weights4 = train(X_train, y_train, weights4, sigmoid, sigmoid_derivative, cross_entropy, epochs=10, batch_size=64, lr=0.01)

    #Task12-E
    print("Task12 - E")
    weights5 = initialize_weights(784, [10], 10)
    trained_weights5 = train(X_train, y_train, weights5, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.01)

    weights6 = initialize_weights(784, [64, 10], 10)
    trained_weights6 = train(X_train, y_train, weights6, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.01)

    weights7 = initialize_weights(784, [128, 10], 10)
    trained_weights7 = train(X_train, y_train, weights7, sigmoid, sigmoid_derivative, mse, epochs=10, batch_size=64, lr=0.01)
