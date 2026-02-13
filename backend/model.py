import numpy as np
import pickle
import os

class ScratchNeuralNet:
    def __init__(self):
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def load_model(self, model_path="../models/mnist_weights.pkl"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, model_path)
            
        with open(full_path, 'rb') as f:
            data = pickle.load(f)
            
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z, temperature=3.0): 
        # Divide by temperature to "soften" the massive weights
        Z = Z / temperature
        expZ = np.exp(Z - np.max(Z, axis=0))
        return expZ / np.sum(expZ, axis=0)

    def predict(self, pixels_list: list) -> dict:
        if self.W1 is None:
            raise ValueError("Model weights not loaded!")

        X = np.array(pixels_list).reshape(784, 1) / 255.0
        
        Z1 = self.W1 @ X + self.b1
        A1 = self.relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        
        A2 = self.softmax(Z2) 
        
        prediction_idx = int(np.argmax(A2, axis=0)[0])
        all_probs = [float(p) for p in A2.flatten().tolist()]
        
        return {
            "predicted_class": prediction_idx, 
            "all_probabilities": all_probs
        }