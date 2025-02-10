# Janus Pro UI

![Janus Pro UI Banner](https://github.com/sbl8/janus-pro-ui/blob/master/sample.png)

Janus Pro UI is a modular application that integrates a FastAPI backend with a React frontend. The backend provides endpoints for image understanding and text-to-image generation using DeepSeek's Janus Pro models, while the frontend (built with React and Vite) serves as the user interface.

## Repository

GitHub: [https://github.com/sbl8/janus-pro-ui](https://github.com/sbl8/janus-pro-ui)

## Project Structure

- **backend/**: Contains the FastAPI server, model loading, configuration, and API endpoints.
- **frontend/**: Contains the React application built with Vite.

## Requirements

### Backend
- Python 3.12+
- NVIDIA GPU with CUDA support
- Required Python packages (see `backend/requirements.txt`)

### Frontend
- Node.js (v14+ recommended)

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sbl8/janus-pro-ui.git
cd janus-pro-ui
```

### 2. Setup the Backend

1. Navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Set environment variables to mitigate CUDA memory fragmentation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```
5. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```
6. The backend will be available at [http://localhost:8080](http://localhost:8080).

### 3. Setup the Frontend

1. Navigate to the `frontend/` directory:
   ```bash
   cd ../frontend
   ```
2. Install the Node.js dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser at [http://localhost:3000](http://localhost:3000) to access the UI.

### 4. Production Build (Frontend)

To build the production version of the frontend:
```bash
npm run build
```
The production assets will be output to the `dist/` directory.

## Technical Details

- **Backend**:  
  - Built with FastAPI, PyTorch, and Transformers.  
  - Model loading and configuration are managed in `backend/app/core/config.py` and `backend/app/core/models.py`.  
  - API endpoints for image understanding and text-to-image generation are defined in `backend/app/api/v1/routers/generation.py`.

- **Frontend**:  
  - Developed with React and Vite.  
  - Communicates with the backend via REST API calls (using axios) and handles streaming responses for image display.  
  - Main source code is located in `frontend/src/`.

## Troubleshooting

- **CUDA Out of Memory**:  
  Adjust parameters such as `parallel_size` or lower the image resolution if you encounter GPU memory errors.

- **API Endpoint Issues**:  
  Ensure that the backend server is running and that the endpoint URLs in the frontend match those defined in the backend.

- **Dependency Problems**:  
  Confirm that all dependencies are installed correctly in both the backend and frontend environments.
