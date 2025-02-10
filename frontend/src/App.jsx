import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(42);
  const [guidance, setGuidance] = useState(7.5);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setError(null);
    setImage(null);

    try {
      // Create a FormData object and append the fields expected by the endpoint
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('seed', seed);
      formData.append('guidance', guidance);

      // Send the form data; set responseType to 'blob' because the endpoint streams image data
      const response = await axios.post('http://127.0.0.1:8080/api/v1/generate_images/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob',
      });

      // Create a URL for the returned image blob and set it for rendering
      const imageUrl = URL.createObjectURL(response.data);
      setImage(imageUrl);
    } catch (err) {
      // If error details are returned as an object, stringify them so React can render a string
      const errorMsg = err.response?.data?.detail;
      setError(typeof errorMsg === 'object' ? JSON.stringify(errorMsg) : errorMsg || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header>
        <h1>Janus Pro UI</h1>
        <p>UI Wrapper for Deepseek's Janus Pro</p>
      </header>
      <main>
        <div className="input-section">
          <textarea
            placeholder="Enter your prompt here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
          />
          <div>
            <label>Seed:</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </div>
          <div>
            <label>Guidance:</label>
            <input
              type="number"
              step="0.1"
              value={guidance}
              onChange={(e) => setGuidance(Number(e.target.value))}
            />
          </div>
          <button onClick={handleGenerate} disabled={loading}>
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
        {image && (
          <div className="image-container">
            <img src={image} alt="Generated" />
          </div>
        )}
      </main>
      <footer>
        <p>&copy; 2025 Janus Pro UI. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
