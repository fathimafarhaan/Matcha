import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

const Logout = ({ setAuth, setRole }) => {
  const navigate = useNavigate();

  useEffect(() => {
    // Clear authentication data
    localStorage.removeItem("token");
    localStorage.removeItem("role");

    // Reset authentication state
    setAuth(false);
    setRole(null);

    // Redirect to login after a short delay
    setTimeout(() => {
      navigate("/login");
    }, 1000);
  }, [navigate, setAuth, setRole]);

  return (
    <div className="logout-container">
      <p>Logging out... Please wait.</p>
    </div>
  );
};

export default Logout;
