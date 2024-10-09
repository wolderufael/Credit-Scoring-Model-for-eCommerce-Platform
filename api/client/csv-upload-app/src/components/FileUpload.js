// // src/components/FileUpload.js
// import React, { useState } from "react";
// import axios from "axios";

// const FileUpload = () => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [jsonResponse, setJsonResponse] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState("");

//   const handleFileChange = (event) => {
//     setSelectedFile(event.target.files[0]);
//   };

//   const handleSubmit = async (event) => {
//     event.preventDefault();
//     if (!selectedFile) {
//       setError("Please select a CSV file.");
//       return;
//     }
//     setLoading(true);
//     setError("");
//     setJsonResponse(null);

//     const formData = new FormData();
//     formData.append("file", selectedFile);

//     try {
//       const response = await axios.post(
//         "http://localhost:5000/predict_logistic_regression",
//         formData,
//         {
//           headers: {
//             "Content-Type": "multipart/form-data",
//           },
//         }
//       );
//       setJsonResponse(response.data);
//     } catch (err) {
//       setError("Failed to upload file. " + err.message);
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div>
//       <h2>Upload CSV File</h2>
//       <form onSubmit={handleSubmit}>
//         <input type="file" accept=".csv" onChange={handleFileChange} />
//         <button type="submit" disabled={loading}>
//           {loading ? "Uploading..." : "Upload and Predict"}
//         </button>
//       </form>
//       {error && <p style={{ color: "red" }}>{error}</p>}
//       {jsonResponse && (
//         <div>
//           <h3>Prediction Result</h3>
//           <pre>{JSON.stringify(jsonResponse, null, 2)}</pre>
//           <a
//             href={`data:text/json;charset=utf-8,${encodeURIComponent(
//               JSON.stringify(jsonResponse)
//             )}`}
//             download="predictions.json"
//           >
//             Download JSON
//           </a>
//         </div>
//       )}
//     </div>
//   );
// };

// export default FileUpload;

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
        "http://localhost:5000/predict_logistic_regression",
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
