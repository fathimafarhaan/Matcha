import React from "react";
import "./ResumeDisplay.css";

const ResumeDisplay = ({ resumeText }) => {
  return (
    <div className="resume-container">
      <h2>Extracted Resume Data</h2>
      <pre>{resumeText}</pre>
    </div>
  );
};

export default ResumeDisplay;