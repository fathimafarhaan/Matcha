import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import FileUpload from "./FileUpload";
import "../styles/RecruiterDashboard.css";

const RecruiterDashboard = () => {
    const [resumeData, setResumeData] = useState(null);
    const [resumeScore, setResumeScore] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const token = localStorage.getItem("token");
        const role = localStorage.getItem("role");
        if (!token || role !== "recruiter") {
            navigate("/login");
        }
    }, [navigate]);

    const handleFileUpload = async (files) => {
        try {
            if (!files || files.length === 0) {
                throw new Error("Please select a file.");
            }

            const formData = new FormData();
            formData.append("file", files[0]);
            
            setLoading(true);
            setError(null);
            setSuccess(null);

            const token = localStorage.getItem("token");
            const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "multipart/form-data"
                },
                onUploadProgress: (progressEvent) => {
                    const percentCompleted = Math.round(
                        (progressEvent.loaded * 100) / progressEvent.total
                    );
                    console.log(`Upload Progress: ${percentCompleted}%`);
                }
            });

            if (response.data?.parsedResume && response.data?.score !== undefined) {
                
                setResumeData({
                    text: response.data.parsedResume,
                    name: response.data.name,
                    phone: response.data.phone,
                    email: response.data.email,
                    experience: response.data.experience,
                    experience_display: response.data.experience_display, // Add this line
                    skills: response.data.skills,
                    education: response.data.education,
                    projects: response.data.projects
                });
                setResumeScore(response.data.score);
                setSuccess("Resume processed successfully!");

            } else {
                throw new Error("Unexpected server response. Please try again.");
            }
        } catch (error) {
            console.error("Upload error:", error);
            setError(error.response?.data?.error || error.message || "Failed to process the resume.");
        } finally {
            setLoading(false);
        }
    };

    const handleSaveCSV = async () => {
        try {
            if (!resumeData || resumeScore === null || resumeScore === undefined) {
                throw new Error("No resume data or resume score available to save.");
            }

            const csvData = {
                name: resumeData.name || "N/A",
                phone: resumeData.phone || "N/A",
                email: resumeData.email || "N/A",
                experience: resumeData.experience_display || "N/A",
                skills: Array.isArray(resumeData.skills) ? resumeData.skills : "N/A",
                education: resumeData.education || "N/A",
                projects: resumeData.projects || "N/A",
                score: resumeScore,
            };
            const token = localStorage.getItem("token");
            await axios.post("http://127.0.0.1:5000/save_csv", csvData, {
                headers: { 
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json"
                }
            });

            setSuccess("CSV file saved successfully!");
        } catch (error) {
            console.error("CSV Save error:", error);
            setError(error.response?.data?.error || error.message || "Failed to save CSV file.");
        }
    };

    const handleLogout = () => {
        localStorage.removeItem("token");
        localStorage.removeItem("role");
        navigate("/login");
    };

    return (
        <div className="dashboard">
            <h1>Recruiter Dashboard</h1>
            <button className="logout-btn" onClick={handleLogout}>Logout</button>
            
            <FileUpload onFileUpload={handleFileUpload} />
            
            {loading && <p className="loading">Processing...</p>}
            {error && <p className="error">{error}</p>}
            {success && <p className="success">{success}</p>}
            
            {resumeData && (
                <div className="resume-container">
                    <div className="resume-header">
                        <h2 className="score">AI Score: {resumeScore} / 100</h2>
                        <button className="csv-btn" onClick={handleSaveCSV}>Save CSV</button>
                    </div>
                    
                    <div className="candidate-info">
                        <h3>Candidate Information</h3>
                        <div className="info-grid">
                            <div className="info-item">
                                <span className="info-label">Name:</span>
                                <span className="info-value">{resumeData.name || "N/A"}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Email:</span>
                                <span className="info-value">{resumeData.email || "N/A"}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Phone:</span>
                                <span className="info-value">{resumeData.phone || "N/A"}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Work Experience:</span>
                                <span className="info-value">{resumeData.experience_display || "N/A"}</span>
                            </div>
                            <div className="info-item">
                                <span className="info-label">Projects:</span>
                                <span className="info-value">{resumeData.projects || "N/A"}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div className="skills-section">
                        <h3>Skills</h3>
                        <div className="skills-list">
                            {Array.isArray(resumeData.skills) && resumeData.skills.length > 0 ? (
                                resumeData.skills.map((skill, index) => (
                                    <div key={index} className="skill-tag">{skill}</div>
                                ))
                            ) : (
                                <p>No skills detected</p>
                            )}
                        </div>
                    </div>
                    
                   
                </div>
            )}
        </div>
    );
};

export default RecruiterDashboard;