from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ImageInput, PredictionOutput
from .model import ScratchNeuralNet
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
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
    if canvas_data is None:
        return {"Please draw a digit": 1.0}

    # 1. Get the raw RGBA image from Gradio
    img_array = canvas_data["composite"]
    img = Image.fromarray(img_array)

    # 2. Safety Net: Create a pure white background
    background = Image.new("RGB", img.size, (255, 255, 255))
    
    # 3. Paste the drawing over the white background (using its own transparency as a mask)
    # This guarantees we have a black drawing on a white background
    if img.mode == 'RGBA':
        background.paste(img, mask=img.split()[3])
    else:
        background.paste(img)

    # 4. Convert to Grayscale
    gray_img = background.convert("L")

    # 5. Invert it! (MNIST expects White ink on a Black background)
    mnist_img = ImageOps.invert(gray_img)

    # 6. Shrink to 28x28
    mnist_img = mnist_img.resize((28, 28))

    # 7. Flatten into our 784 numbers
    pixels = np.array(mnist_img).flatten().tolist()

    try:
        response = requests.post(
            "http://127.0.0.1:7860/predict",
            json={"pixels": pixels}
        )

        if response.status_code == 200:
            result = response.json()
            if "all_probabilities" in result:
                probs = result["all_probabilities"]
                confidence_dict = {str(i): float(probs[i]) for i in range(10)}
                return confidence_dict
            else:
                return {"Error: Check API Schema": 1.0}
        else:
            return {"API Error": 1.0}

    except Exception:
        return {"Connection Error": 1.0}

ui = gr.Interface(
    fn=process_drawing,
    inputs=gr.Sketchpad(label="Draw a digit (0-9) here"),
    outputs=gr.Label(num_top_classes=3, label="AI Confidence Level 🧠"),
    title="🔢 Scratch Neural Network",
    description="A custom 2-layer neural network built entirely with NumPy. <br><br> **[👉 Click here to view the Developer API Docs](/docs)**",
    flagging_mode="never"
)

app = gr.mount_gradio_app(app, ui, path="/")