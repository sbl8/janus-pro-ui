# Backend - Janus Pro UI

This directory contains the FastAPI backend for Janus Pro UI. It handles model loading, API endpoint definitions, and inference logic for both image understanding and text-to-image generation using DeepSeek's Janus Pro models.

## Directory Structure

- **app/main.py**: The application entry point that configures the FastAPI app and loads models during startup.
- **app/api/v1/routers/generation.py**: Contains all the API routes for image understanding (`/understand_image_and_question/`) and image generation (`/generate_images/`).
- **app/core/config.py**: Project configuration, including model settings and CORS origins.
- **app/core/models.py**: Contains the function to load and configure the Janus model.
- **app/schemas.py**: Pydantic schemas for request and response validation.

## Setup Instructions

1. Navigate to the `backend/` directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Set environment variables such as:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
5. Run the backend server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```

## Technical Details

- **Model Loading**: The Janus Pro model and processor are loaded using settings from `app/core/config.py`. The model is configured for low CPU memory usage and loaded onto the GPU.
- **API Endpoints**: Defined in `app/api/v1/routers/generation.py`, these endpoints expect Form data for parameters and provide JSON responses or image streams.
- **Memory Management**: The code calls `torch.cuda.empty_cache()` and uses garbage collection (`gc.collect()`) to help manage GPU memory.

## Troubleshooting

- Check logs for CUDA OOM errors and adjust parameters like `parallel_size` if needed.
- Validate that the environment’s GPU memory is not consumed by other processes.

This backend is the core engine powering the Janus Pro UI.
