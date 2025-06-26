import React, { useState } from "react";
import { useDropzone } from "react-dropzone";

const FileUpload = ({ onFileUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [error, setError] = useState(null);

  const { getRootProps, getInputProps } = useDropzone({
    accept: ".pdf,.docx",
    maxFiles: 1, // Allow only one file at a time
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length === 0) {
        setError("Invalid file type. Please upload a PDF or DOCX file.");
        return;
      }

      setSelectedFile(acceptedFiles[0].name);
      setError(null);
      onFileUpload(acceptedFiles);
    },
    onDropRejected: () => {
      setError("Unsupported file type. Please upload a PDF or DOCX.");
    },
  });

  return (
    <div>
      <div
        {...getRootProps()}
        style={{
          border: "2px dashed #007bff",
          padding: "20px",
          textAlign: "center",
          borderRadius: "8px",
          cursor: "pointer",
          backgroundColor: "#f8f9fa",
          transition: "background-color 0.2s ease-in-out",
        }}
      >
        <input {...getInputProps()} />
        <p>Drag & drop a resume file here, or click to select a file</p>
      </div>

      {selectedFile && <p style={{ marginTop: "10px", fontWeight: "bold" }}>Selected File: {selectedFile}</p>}
      {error && <p style={{ color: "red", marginTop: "10px" }}>{error}</p>}
    </div>
  );
};

export default FileUpload;