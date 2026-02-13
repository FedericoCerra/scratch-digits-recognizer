import numpy as np
import pickle
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
# DATA PREPARATION 

def OneHotEncode(y):
    y = y.astype(int)
    one_hot = np.zeros((y.size, y.max() + 1))
    one_hot[np.arange(y.size), y] = 1.0
    return one_hot

def load_and_prep_data():
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    
    X = np.asarray(X) / 255.0
    
    y_encoded = OneHotEncode(y).T
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded.T, test_size=0.2, random_state=42
    )
    
    X_train = X_train.T
    y_train = y_train.T
    X_test = X_test.T
    y_test = y_test.T
    
    print(f"Training on {X_train.shape[1]} examples.")
    return X_train, y_train, X_test, y_test

# MATH FUNCTIONS

def init_parameters(input_size, hidden_size, output_size):
    W1 = (np.random.rand(hidden_size, input_size) - 0.5) * 0.6
    b1 = np.zeros((hidden_size, 1))
    W2 = (np.random.rand(output_size, hidden_size) - 0.5) * 0.6
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def ReLUDerivative(Z):
    return Z > 0

def stable_softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forwardPropagation(X, W1, b1, W2, b2):
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = stable_softmax(Z2)
    return Z1, A1, Z2, A2

def backwards_prop(X, y, Z1, A1, A2, W2):
    m = X.shape[1]
    dZ2 = A2 - y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = W2.T @ dZ2 * ReLUDerivative(Z1)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, y):
    return np.mean(predictions == y)

# TRAINING PIPELINE

def train_network(X, y, alpha=0.4, iterations=260):
    print("Initializing parameters...")
    W1, b1, W2, b2 = init_parameters(X.shape[0], 10, y.shape[0])
    
    print(f"Starting gradient descent ({iterations} iterations, alpha={alpha})...")
    for i in range(iterations):
        # Forward pass
        Z1, A1, Z2, A2 = forwardPropagation(X, W1, b1, W2, b2)
        
        # Backward pass
        dW1, db1, dW2, db2 = backwards_prop(X, y, Z1, A1, A2, W2)
        
        # Update weights
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Print progress every 10 iterations
        if i % 10 == 0:
            preds = get_predictions(A2)
            actual_y = np.argmax(y, axis=0)
            acc = get_accuracy(preds, actual_y)
            print(f"Iteration {i:3} | Training Accuracy: {acc:.2%}")
            
    return W1, b1, W2, b2

def save_model(W1, b1, W2, b2, filename="mnist_weights.pkl"):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    models_dir = os.path.join(current_dir, "../models")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 4. Create the full file path
    full_path = os.path.join(models_dir, filename)

    model_data = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    
    with open(full_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model weights successfully saved to {filename}")

# MAIN 

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_prep_data()
    
    W1, b1, W2, b2 = train_network(X_train, y_train, alpha=0.4, iterations=3000)
    
    print("\nEvaluating on Test Set...")
    _, _, _, A2_test = forwardPropagation(X_test, W1, b1, W2, b2)
    y_pred_test = get_predictions(A2_test)
    y_actual_test = np.argmax(y_test, axis=0)
    
    test_accuracy = get_accuracy(y_pred_test, y_actual_test)
    print(f"Final Test Accuracy: {test_accuracy:.2%}\n")
    
    save_model(W1, b1, W2, b2, "mnist_weights.pkl")