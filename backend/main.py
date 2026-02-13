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

app = FastAPI(
    title="Scratch Neural Network Digit Recognizer", 
    description="Welcome to the API! **[⬅️ Click here to go back to the Drawing Canvas](/)**",
    lifespan=lifespan
)
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
        return {"Please draw a digit": 1.0}
        
    img_array = canvas_data["composite"]
    img = Image.fromarray(img_array).convert("L").resize((28, 28))
    pixels = np.array(img).flatten().tolist()
    
    try:
        response = requests.post("http://127.0.0.1:7860/predict", json={"pixels": pixels})
        if response.status_code == 200:
            result = response.json()
            
            # Check if your API returns a list of probabilities for all 10 digits
            if "probabilities" in result:
                probs = result["probabilities"]
                # Zip the digits "0"-"9" with their percentage confidences
                confidence_dict = {str(i): float(probs[i]) for i in range(10)}
                return confidence_dict
            else:
                # Fallback: If your API only returns the winning class, just show it at 100%
                winning_class = str(result['predicted_class'])
                return {f"Digit {winning_class}": 1.0}
        else:
            return {"API Error": 1.0}
    except Exception as e:
        return {"Connection Error": 1.0}

# Create the Gradio interface
ui = gr.Interface(
    fn=process_drawing,
    inputs=gr.Sketchpad(label="Draw a digit (0-9) here"), 
    outputs=gr.Label(num_top_classes=3, label="AI Confidence Level 🧠"), # <-- THIS IS THE UPGRADE
    title="🔢 Scratch Neural Network",
    description="A custom 2-layer neural network built entirely with NumPy. <br><br> **[👉 Click here to view the Developer API Docs](/docs)**",
    flagging_mode="never"
)

app = gr.mount_gradio_app(app, ui, path="/")