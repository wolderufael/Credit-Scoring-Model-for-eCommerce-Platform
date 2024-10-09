import React, { useState } from "react";
import axios from "axios";
import "./FileUpload.css"; // Import CSS file

const UploadForm = () => {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPredictions(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        // "http://localhost:5000/predict_logistic_regression",
        "https://credit-scoring-model-for-ecommerce.onrender.com/predict_logistic_regression",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      setPredictions(response.data.predictions);
    } catch (err) {
      setError("There was an error processing your request.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Logistic Regression Prediction</h1>
      <form onSubmit={handleSubmit} className="upload-form">
        <label className="file-label">
          <input
            type="file"
            onChange={handleFileChange}
            className="file-input"
          />
          <span>Choose a CSV file</span>
        </label>
        <button type="submit" className="submit-btn" disabled={loading}>
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {predictions && (
        <div className="predictions">
          <h2>Predictions</h2>
          <ul>
            {predictions.map((pred, index) => (
              <li key={index}>
                Prediction {index + 1}: {pred}
              </li>
            ))}
          </ul>
          <button className="download-btn">Download JSON</button>
        </div>
      )}
    </div>
  );
};

export default UploadForm;
