import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../styles/AuthForms.css';
import socket from '../socket'; // adjust path if your socket.js is elsewhere


export default function RegisterPage() {
  const navigate = useNavigate();

  const [form, setForm] = useState({
    username: '',
    password: '',
    phone: '',
    email: '',
    role: 'doctor',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prevForm) => ({
      ...prevForm,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post(`${process.env.REACT_APP_API_URL}/api/register`, form);
      alert('Registration successful');
      socket.onopen = () => {
        console.log('✅ WebSocket connected after registration');
        socket.send('New user registered!');
      };

      socket.onmessage = (event) => {
        console.log('📨 Message from server:', event.data);
      };

      navigate('/login');
    } catch (err) {
      console.error(err.response?.data || err.message);
      alert('Registration failed');
    }
  };

  return (
    <div className="auth-container">
      <h2 className="form-title">Create an Account</h2>
      <form className="auth-form" onSubmit={handleSubmit}>
        <input
          name="username"
          type="text"
          placeholder="Username"
          value={form.username}
          onChange={handleChange}
          required
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          value={form.password}
          onChange={handleChange}
          required
        />
        <input
          name="phone"
          type="tel"
          placeholder="Phone Number"
          value={form.phone}
          onChange={handleChange}
          required
        />
        <input
          name="email"
          type="email"
          placeholder="Email"
          value={form.email}
          onChange={handleChange}
          required
        />
        <select
          name="role"
          value={form.role}
          onChange={handleChange}
          required
        >
          <option value="doctor">Doctor</option>
          <option value="scientist">Scientist</option>
        </select>
        <button type="submit" className="submit-btn">
          Register
        </button>
      </form>
    </div>
  );
}
