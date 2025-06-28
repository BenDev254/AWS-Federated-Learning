// src/pages/LandingPage.js
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import socket from '../socket'; //

export default function LandingPage() {
  const navigate = useNavigate();
  const [wsMessages, setWsMessages] = useState([]);


   useEffect(() => {
    socket.onopen = () => {
      console.log('🟢 WebSocket connected (Landing)');
      socket.send('Landing page connected');
    };

    socket.onmessage = (event) => {
      console.log('📨 WS Message (Landing):', event.data);
      setWsMessages((prev) => [...prev, event.data]);
    };

    return () => {
      socket.onmessage = null;
    };
  }, []);

  

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Federated Learning Environment</h1>
      <p style={styles.subtitle}>by Techlife Collective</p>
      <div style={styles.buttonGroup}>
        <button style={styles.button} onClick={() => navigate('/register')}>Register</button>
        <button style={styles.button} onClick={() => navigate('/login')}>Login</button>
      </div>
      {wsMessages.length > 0 && (
        <div style={styles.wsLog}>
          <h4>Live Server Messages</h4>
          <ul style={styles.logList}>
            {wsMessages.map((msg, idx) => (
              <li key={idx}>{msg}</li>
            ))}
          </ul>
        </div>
      )}
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
    backgroundColor: '#f9f9f9',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: '#555',
    marginBottom: '2rem',
  },
  buttonGroup: {
    display: 'flex',
    gap: '1rem',
  },
  button: {
    padding: '10px 20px',
    fontSize: '1rem',
    cursor: 'pointer',
    border: '1px solid #ccc',
    borderRadius: '5px',
    backgroundColor: '#fff',
  }
};
