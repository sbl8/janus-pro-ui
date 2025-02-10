# Frontend - Janus Pro UI

This directory contains the React-based frontend for Janus Pro UI. The frontend provides a user interface to interact with the FastAPI backend, allowing users to enter prompts and view generated images.

## Directory Structure

- **src/App.jsx**: The main React component that handles user input and displays generated images.
- **src/App.css**: Styling for the application.
- **src/main.jsx**: The application’s entry point.
- **vite.config.js**: Vite configuration for fast development and hot module replacement.

## Setup Instructions

1. Navigate to the `frontend/` directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
4. Open your browser at [http://localhost:3000](http://localhost:3000) to interact with the UI.

## Technical Details

- **Communication with Backend**: The frontend uses axios to POST Form data to the backend endpoint (`/api/v1/generate_images/`). It expects a streamed image response, which is then rendered in the UI.
- **Form Handling**: The UI allows users to input a text prompt and adjust parameters (such as seed and guidance). These are sent as FormData.
- **Error Handling**: If the backend returns an error (e.g., validation or server error), the frontend converts error details to a string and displays them.

## Troubleshooting

- **Endpoint Errors (422/500)**: Ensure that the backend is running and that the payload fields match what the API expects.
- **Image Streaming**: The frontend handles the streaming response by creating object URLs for received image blobs.

This frontend provides a responsive and user-friendly interface for interacting with DeepSeek’s Janus Pro model.
