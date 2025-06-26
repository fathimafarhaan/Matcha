import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "../styles/AdminDashboard.css";

const AdminDashboard = () => {
  const [scoringCriteria, setScoringCriteria] = useState({
    experienceWeight: 0,
    projectsWeight: 0,
    skillsWeight: 0,
    educationWeight: 0,
  });

  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [success, setSuccess] = useState(null);
  const navigate = useNavigate();

  // ✅ Fetch Scoring Criteria
  const fetchScoringCriteria = useCallback(async () => {
    setLoading(true);
    const token = localStorage.getItem("token");

    if (!token) {
      navigate("/login");
      return;
    }

    try {
      const response = await axios.get("http://127.0.0.1:5000/admin/get_scoring_criteria", {
        headers: { Authorization: `Bearer ${token}` },
      });

      setScoringCriteria({
        experienceWeight: response.data.experience_weight,
        projectsWeight: response.data.projects_weight,
        skillsWeight: response.data.skills_weight,
        educationWeight: response.data.education_weight,
      });
    } catch (error) {
      console.error("Error fetching criteria:", error);
      setError(error.response?.data?.message || "Failed to fetch scoring criteria.");
    } finally {
      setLoading(false);
    }
  }, [navigate]);

  useEffect(() => {
    fetchScoringCriteria();
  }, [fetchScoringCriteria]);

  // ✅ Update Scoring Criteria
  const handleUpdateCriteria = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    const token = localStorage.getItem("token");

    try {
      await axios.post(
        "http://127.0.0.1:5000/admin/update_scoring",
        {
          experience_weight: parseFloat(scoringCriteria.experienceWeight),
          projects_weight: parseFloat(scoringCriteria.projectsWeight),
          skills_weight: parseFloat(scoringCriteria.skillsWeight),
          education_weight: parseFloat(scoringCriteria.educationWeight),
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      setSuccess("Scoring criteria updated successfully!");
      fetchScoringCriteria();
    } catch (error) {
      console.error("Error updating criteria:", error);
      setError(error.response?.data?.message || "Failed to update scoring criteria.");
    }
  };

  // ✅ CSV Download Feature
  const handleDownloadCSV = async () => {
    const token = localStorage.getItem("token");

    try {
      const response = await axios.get("http://127.0.0.1:5000/admin/download_csv", {
        headers: { Authorization: `Bearer ${token}` },
        responseType: "blob",
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "resumes.csv");
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Error downloading CSV:", error);
      setError("Failed to download CSV file.");
    }
  };

  // ✅ Logout Function
  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  if (loading) return <p>Loading...</p>;

  return (
    <div className="dashboard">
      <h1>Admin Dashboard</h1>
      <button className="logout-btn" onClick={handleLogout}>Logout</button>

      <button onClick={handleDownloadCSV} className="download-btn">
        Download CSV File
      </button>

      <h2>Edit Scoring Criteria</h2>
      {error && <p className="error">{error}</p>}
      {success && <p className="success">{success}</p>}

      <form onSubmit={handleUpdateCriteria}>
        <label>
          Experience Weight:
          <input
            type="number"
            step="0.1"
            value={scoringCriteria.experienceWeight || ""}
            onChange={(e) =>
              setScoringCriteria({ ...scoringCriteria, experienceWeight: e.target.value })
            }
          />
        </label>
        <label>
          Projects Weight:
          <input
            type="number"
            step="0.1"
            value={scoringCriteria.projectsWeight || ""}
            onChange={(e) =>
              setScoringCriteria({ ...scoringCriteria, projectsWeight: e.target.value })
            }
          />
        </label>
        <label>
          Skills Weight:
          <input
            type="number"
            step="0.1"
            value={scoringCriteria.skillsWeight || ""}
            onChange={(e) =>
              setScoringCriteria({ ...scoringCriteria, skillsWeight: e.target.value })
            }
          />
        </label>
        <label>
          Education Weight:
          <input
            type="number"
            step="0.1"
            value={scoringCriteria.educationWeight || ""}
            onChange={(e) =>
              setScoringCriteria({ ...scoringCriteria, educationWeight: e.target.value })
            }
          />
        </label>

        <button type="submit">Update Criteria</button>
      </form>
    </div>
  );
};

export default AdminDashboard;
