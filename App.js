import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Route, Routes, Navigate } from "react-router-dom";
import Login from "./components/Login";
import RecruiterDashboard from "./components/RecruiterDashboard";
import AdminDashboard from "./components/AdminDashboard";
import Logout from "./components/Logout";

function App() {
  const [auth, setAuth] = useState(false);
  const [role, setRole] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem("token");
    const userRole = localStorage.getItem("role");

    if (token && userRole) {
      setAuth(true);
      setRole(userRole);
    } else {
      setAuth(false);
      setRole(null);
    }
  }, []);

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<Login setAuth={setAuth} setRole={setRole} />} />
        
        {/* Recruiter Dashboard Route */}
        <Route
          path="/recruiter-dashboard"
          element={auth && role === "recruiter" ? <RecruiterDashboard /> : <Navigate to="/login" replace />}
        />
        
        {/* Admin Dashboard Route */}
        <Route
          path="/admin-dashboard"
          element={auth && role === "admin" ? <AdminDashboard /> : <Navigate to="/login" replace />}
        />
        
        {/* Logout Route */}
        <Route path="/logout" element={<Logout setAuth={setAuth} setRole={setRole} />} />
        
        {/* Catch-all Route Redirects to Login */}
        <Route path="*" element={<Navigate to={auth ? (role === "admin" ? "/admin-dashboard" : "/recruiter-dashboard") : "/login"} replace />} />
      </Routes>
    </Router>
  );
}

export default App;