// src/components/Login.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

export default function LoginPage() {
  const [form, setForm] = useState({ username: '', password: '' });
  const navigate = useNavigate();

  const handleChange = (e) =>
    setForm({ ...form, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const res = await axios.post('http://localhost:8000/api/login', form, {
        headers: { 'Content-Type': 'application/json' },
      });

      const { token, role } = res.data;

      localStorage.setItem('token', token);
      localStorage.setItem('role', role);

      if (role === 'doctor') {
        navigate('/doctor');
      } else if (role === 'scientist') {
        navigate('/scientist');
      } else {
        alert('Unknown role');
        navigate('/');
      }
    } catch (err) {
      alert('Login failed');
      console.error(err.response?.data || err.message);
    }
  };

  const handleRedirectToRegister = () => {
    navigate('/register');
  };

  return (
    <div style={styles.container}>
      <h2>Login</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          name="username"
          placeholder="Username"
          onChange={handleChange}
          required
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          onChange={handleChange}
          required
        />
        <div style={styles.buttonGroup}>
          <button type="submit" style={styles.button}>Login</button>
          <button type="button" onClick={handleRedirectToRegister} style={styles.button}>
            Register
          </button>
        </div>
      </form>
    </div>
  );
}
const styles = {
  container: {
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    fontFamily: 'sans-serif',
    padding: '2rem',
  },
  form: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center', // Center contents horizontally
    gap: '1rem',
    maxWidth: '300px',
    width: '100%',
  },
  buttonGroup: {
    display: 'flex',
    justifyContent: 'center',
    gap: '1rem',
    width: '100%', // Ensure full width to center children properly
  },
  button: {
    flex: 1, // Allow buttons to share space evenly
    padding: '0.5rem 1rem',
    cursor: 'pointer',
    border: '1px solid #ccc',
    backgroundColor: '#f0f0f0',
    borderRadius: '4px',
    maxWidth: '120px', // Optional: keeps buttons from stretching too far
  },
};
