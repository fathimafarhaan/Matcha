import React, { useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./Login.css";

const Login = ({ setAuth, setRole }) => {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("token");
    const role = localStorage.getItem("role");

    if (token && role) {
      const verifyToken = async () => {
        try {
          await axios.post("http://localhost:5000/verify-token", {}, {
            headers: { Authorization: `Bearer ${token}` },
          });
          navigate(role === "admin" ? "/admin-dashboard" : "/recruiter-dashboard");
        } catch (error) {
          console.warn("🔴 Invalid token. Logging out...");
          localStorage.removeItem("token");
          localStorage.removeItem("role");
        }
      };
      verifyToken();
    }
  }, [navigate]);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/login", {
        username,
        password,
      });

      const { token, role } = response.data;
      localStorage.setItem("token", token);
      localStorage.setItem("role", role);
      setAuth(true);
      setRole(role);

      navigate(role === "admin" ? "/admin-dashboard" : "/recruiter-dashboard");
    } catch (err) {
      console.error("❌ Login error:", err.response?.data || err.message);
      if (!err.response) {
        setError("Cannot connect to the server. Please check your network.");
      } else if (err.response.status === 401) {
        setError("Invalid username or password.");
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h1 className="project-name">Resume Parser</h1>
      <h2>Login</h2>
      {error && <p className="error">{error}</p>}
      <form onSubmit={handleLogin}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          autoFocus
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? "Logging in..." : "Login"}
        </button>
      </form>
    </div>
  );
};

export default Login;
