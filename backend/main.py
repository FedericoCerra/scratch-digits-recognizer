from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ImageInput, PredictionOutput
from .model import ScratchNeuralNet
import gradio as gr
import numpy as np
from PIL import Image
import requests

nn = ScratchNeuralNet()
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Neural Network weights...")
    nn.load_model()
    print("Model loaded successfully!")        
    yield 

app = FastAPI(title="Scratch Neural Network Digit Recognizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to make requests
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionOutput)
def predict_digit(data: ImageInput):
    if nn is None:
        raise HTTPException(status_code=500, detail="Model is not available.")
        
    result_dict = nn.predict(data.pixels)
    
    return PredictionOutput(**result_dict)

def process_drawing(canvas_data):
    """Takes the drawing from Gradio, resizes it, and sends it to our own API."""
    if canvas_data is None:
        return "Please draw a digit."
        
    # Gradio's Sketchpad sends a dict. The 'composite' key holds the RGBA image array.
    img_array = canvas_data["composite"]
    
    # Convert to grayscale ("L") and shrink to 28x28 to match MNIST
    img = Image.fromarray(img_array).convert("L").resize((28, 28))
    
    # Flatten the 28x28 image into a single list of 784 pixels
    pixels = np.array(img).flatten().tolist()
    
    try:
        response = requests.post("http://127.0.0.1:7860/predict", json={"pixels": pixels})
        if response.status_code == 200:
            result = response.json()
            return f"🧠 I think it's a: {result['predicted_class']}"
        else:
            return "API Error: Could not process image."
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

# Create the Gradio interface
ui = gr.Interface(
    fn=process_drawing,
    inputs=gr.Sketchpad(label="Draw a digit (0-9) here"), 
    outputs=gr.Text(label="Network Prediction"),
    title="🔢 Scratch Neural Network",
    description="A custom 2-layer neural network built entirely with NumPy. Draw a digit below!"
)

# Mount the Gradio app onto your existing FastAPI app at the root URL ("/")
app = gr.mount_gradio_app(app, ui, path="/")