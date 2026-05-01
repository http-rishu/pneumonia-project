import React, { useState, useCallback } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  // Handle file selection
  const handleFile = (file) => {
    if (file) {
      // Validate file type
      const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/webp"];
      if (!validTypes.includes(file.type)) {
        setError("Please upload a valid image file (JPG, PNG, BMP, or WebP)");
        return;
      }
      
      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError("File size must be less than 10MB");
        return;
      }

      setImage(file);
      setError(null);
      setResult(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  // Handle drag events
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  // Handle drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  // Handle input change
  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  // Handle upload/submit
  const handleSubmit = async () => {
    if (!image) {
      setError("Please select an X-ray image first");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Failed to connect to server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Reset everything
  const handleReset = () => {
    setImage(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  // Get result color class
  const getResultClass = () => {
    if (!result) return "";
    return result.prediction === "PNEUMONIA" ? "result-pneumonia" : "result-normal";
  };

  // Get severity color
  const getSeverityColor = () => {
    if (!result) return "";
    switch (result.severity) {
      case "Mild": return "#22c55e";
      case "Moderate": return "#f59e0b";
      case "Severe": return "#ef4444";
      default: return "#6b7280";
    }
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="logo">🩺</div>
          <h1>Pneumonia Detection</h1>
          <p className="subtitle">AI-Powered X-Ray Analysis</p>
        </header>

        <main className="main-content">
          {/* Upload Section */}
          {!result && (
            <div className="upload-section">
              <div
                className={`drop-zone ${dragActive ? "drag-active" : ""} ${preview ? "has-image" : ""}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {preview ? (
                  <div className="preview-container">
                    <img src={preview} alt="X-ray preview" className="preview-image" />
                    <button className="remove-btn" onClick={handleReset} title="Remove image">
                      ✕
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon">📤</div>
                    <p className="upload-text">Drag & drop your X-ray image here</p>
                    <p className="upload-subtext">or</p>
                    <label className="browse-btn">
                      Browse Files
                      <input
                        type="file"
                        accept="image/jpeg,image/jpg,image/png,image/bmp,image/webp"
                        onChange={handleChange}
                        hidden
                      />
                    </label>
                    <p className="upload-hint">Supported: JPG, PNG, BMP, WebP (max 10MB)</p>
                  </>
                )}
              </div>

              {error && (
                <div className="error-message">
                  <span>⚠️</span> {error}
                </div>
              )}

              {image && !result && (
                <button 
                  className="analyze-btn" 
                  onClick={handleSubmit}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Analyzing...
                    </>
                  ) : (
                    <>🔍 Analyze X-Ray</>
                  )}
                </button>
              )}
            </div>
          )}

          {/* Result Section */}
          {result && (
            <div className="result-section">
              <div className="result-card">
                <div className="result-header">
                  <div className={`result-badge ${getResultClass()}`}>
                    {result.prediction === "PNEUMONIA" ? "🦠" : "✅"}
                    <span>{result.prediction}</span>
                  </div>
                </div>

                <div className="result-details">
                  <div className="detail-item">
                    <span className="detail-label">Confidence</span>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${result.confidence}%` }}
                      ></div>
                    </div>
                    <span className="detail-value">{result.confidence}%</span>
                  </div>

                  {result.prediction === "PNEUMONIA" && (
                    <div className="detail-item">
                      <span className="detail-label">Severity</span>
                      <span 
                        className="detail-value severity"
                        style={{ color: getSeverityColor() }}
                      >
                        {result.severity}
                      </span>
                    </div>
                  )}

                  <p className="result-message">{result.message}</p>
                </div>

                <button className="new-analysis-btn" onClick={handleReset}>
                  🔄 New Analysis
                </button>
              </div>
            </div>
          )}
        </main>

        <footer className="footer">
          <p>Powered by Deep Learning • For medical reference only</p>
        </footer>
      </div>
    </div>
  );
}

export default App;