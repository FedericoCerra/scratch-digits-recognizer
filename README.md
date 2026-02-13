---
title: Scratch Digits Recognizer
emoji: 🔢
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---
# Scratch Neural Network Digit Recognizer

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Deployed%20on-Hugging%20Face-ffcc00.svg)](https://huggingface.co/spaces/fedede/digits-recognizer)
![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00?style=flat)
![NumPy](https://img.shields.io/badge/Math-NumPy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

A full-stack machine learning project featuring a **2-layer neural network built entirely from scratch** (using only NumPy), wrapped in a **FastAPI** REST service, paired with a native **Gradio** web interface, and fully **containerized with Docker**.

### 🚀 [Click here to try the Live Web App on Hugging Face!](https://huggingface.co/spaces/fedede/digits-recognizer)

---

## Overview

This project demonstrates the internal mechanics of deep learning by implementing backpropagation and gradient descent without the use of high-level frameworks like PyTorch or TensorFlow.

* **The Brain:** A custom-built MLP (Multi-Layer Perceptron) trained on the MNIST dataset.
* **The Bridge:** A FastAPI backend providing high-performance inference.
* **The UI:** A Gradio frontend natively mounted to the FastAPI app for live browser drawing.
* **The Box:** A Docker configuration for consistent deployment across any environment.

---

## Tech Stack

* **Core Math:** NumPy (Matrix multiplication, Activation functions, Backprop)
* **API Framework:** FastAPI
* **Frontend UI:** Gradio
* **Image Processing:** Pillow (PIL)
* **Data Validation:** Pydantic
* **Deployment:** Docker & Hugging Face Spaces
* **Environment:** Python 3.11+

---

## Project Structure

```text
.
├── backend/
│   ├── main.py          # FastAPI application, Gradio UI & lifespan management
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

* Visit `http://127.0.0.1:8000/` to draw on the Gradio Canvas.
* Visit `http://127.0.0.1:8000/docs` to interact with the API via Swagger UI.

### Docker Setup

```bash
# Build the image
docker build -t digit-recognizer .

# Run the container
docker run -p 8000:7860 digit-recognizer
```
## Neural Network Details

The network uses a classic feed-forward architecture:
* **Input Layer:** 784 neurons (28x28 flattened grayscale pixels).
* **Hidden Layer:** 10 neurons with **ReLU** activation.
* **Output Layer:** 10 neurons (representing digits 0-9) with **Softmax** activation and Temperature Scaling to smooth probability distributions.

### Mathematical Implementation
The project includes a manual implementation of the following concepts:
* **Forward Propagation:** $$Z = W \cdot X + b$$
* **Cost Function:** Cross-Entropy Loss.
* **Backward Propagation:** Manual derivation of gradients for weights ($W$) and biases ($b$).
* **Optimization:** Stochastic Gradient Descent (SGD).
* **Temperature Scaling Softmax:** $$Softmax(z_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

---

## Architecture Limitations (Why MLPs struggle with Web Canvas Data)

While this model achieves high accuracy on the MNIST test set, real-world inference via a web canvas exposes the fundamental limitations of standard Multi-Layer Perceptrons compared to modern Convolutional Neural Networks (CNNs):

1. **Spatial Blindness:** By flattening the 28x28 image into a 1D array of 784 pixels, the network loses all spatial awareness (up, down, left, right). Shifting a drawing just a few pixels completely changes the input vector, causing misclassifications.
2. **The Centering Problem:** The MNIST dataset is heavily preprocessed (bounding-box centered and normalized). Freeform digital drawings on a web canvas lack this perfect centering, leading to out-of-distribution data for the model.
3. **Stroke Thickness:** Digital HTML5 canvas brushes lack the natural anti-aliasing and varied pressure of the physical pens used in the original MNIST dataset.

---

## Public API

This project is live on **Hugging Face Spaces**. You can send `POST` requests to the `/predict` endpoint with a JSON body containing a flattened list of 784 pixel values (normalized between 0-255).
