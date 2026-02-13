# Scratch Neural Network Digit Recognizer

A full-stack machine learning project featuring a **2-layer neural network built entirely from scratch** (using only NumPy), wrapped in a **FastAPI** REST service, and fully **containerized with Docker**.

---

## Overview

This project demonstrates the internal mechanics of deep learning by implementing backpropagation and gradient descent without the use of high-level frameworks like PyTorch or TensorFlow.

* **The Brain:** A custom-built MLP (Multi-Layer Perceptron) trained on the MNIST dataset.
* **The Bridge:** A FastAPI backend providing high-performance inference.
* **The Box:** A Docker configuration for consistent deployment across any environment.



---

## Tech Stack

* **Core Math:** NumPy (Matrix multiplication, Activation functions, Backprop)
* **API Framework:** FastAPI
* **Data Validation:** Pydantic
* **Deployment:** Docker & Hugging Face Spaces
* **Environment:** Python 3.11+

---

## Project Structure

```text
.
├── backend/
│   ├── main.py          # FastAPI application & lifespan management
│   ├── model.py         # The Scratch Neural Network class
│   ├── schemas.py       # Pydantic data models
│   ├── train.py         # Training script for weights generation
│   └── requirements.txt # Python dependencies
├── models/
│   └── mnist_weights.pkl # Saved weights (The "Brain")
├── Dockerfile           # Containerization instructions
└── .gitignore           # Keeps the repo clean
```

## Quick Start
### Local Setup
Ensure you have a virtual environment active, then run:
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Visit http://127.0.0.1:8000/docs to interact with the API via Swagger UI.

### Docker Setup

```bash
# Build the image
docker build -t digit-recognizer .

# Run the container
docker run -p 8000:80 digit-recognizer
```

## Neural Network Details

The network uses a classic feed-forward architecture:
* **Input Layer:** 784 neurons (28x28 flattened grayscale pixels).
* **Hidden Layer:** 10 neurons with **ReLU** activation.
* **Output Layer:** 10 neurons (representing digits 0-9) with **Softmax** activation.



### Mathematical Implementation
The project includes a manual implementation of the following concepts:
* **Forward Propagation:** $$Z = W \cdot X + b$$
* **Cost Function:** Cross-Entropy Loss.
* **Backward Propagation:** Manual derivation of gradients for weights ($W$) and biases ($b$).
* **Optimization:** Stochastic Gradient Descent (SGD).



---

## Public API

This project is live on **Hugging Face Spaces**. You can send `POST` requests to the `/predict` endpoint with a JSON body containing a flattened list of 784 pixel values (normalized between 0-255).


