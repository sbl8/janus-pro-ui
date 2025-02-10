# app/schemas.py
from typing import Optional
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    temperature: PositiveFloat = Field(1.0, description="Sampling temperature; must be positive")
    parallel_size: PositiveInt = Field(16, description="Parallel size for generation")
    cfg_weight: PositiveFloat = Field(5.0, description="Classifier-free guidance weight")
    image_token_num_per_image: PositiveInt = Field(576, description="Number of tokens to generate per image")
    img_size: PositiveInt = Field(384, description="Size (height/width) of the output image")
    patch_size: PositiveInt = Field(16, description="Patch size used in image generation")
    seed: Optional[int] = Field(None, description="Optional random seed for reproducibility")

class GenerationResponse(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG image")
