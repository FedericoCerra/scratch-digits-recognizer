from pydantic import BaseModel,Field
from typing import List

class ImageInput(BaseModel):
    pixels : List[float] = Field(..., min_length=784, max_length=784, description="A list of 784 pixel values (28x28 image flattened)")
    
class PredictionOutput(BaseModel):
    predicted_class: int = Field(..., description="The predicted class label (0-9)")
    all_probabilities: List[float] = Field(..., min_length=10, max_length=10, description="A list of probabilities for each class (length 10)")