from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import ImageInput, PredictionOutput
from .model import ScratchNeuralNet


nn = ScratchNeuralNet()
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Neural Network weights...")
    nn.load_model()
    print("Model loaded successfully!")        
    yield 

app = FastAPI(title="Scratch NN Digit Recognizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all websites to make requests
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, etc.
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=PredictionOutput)
def predict_digit(data: ImageInput):
    if nn is None:
        raise HTTPException(status_code=500, detail="Model is not available.")
        
    result_dict = nn.predict(data.pixels)
    
    return PredictionOutput(**result_dict)