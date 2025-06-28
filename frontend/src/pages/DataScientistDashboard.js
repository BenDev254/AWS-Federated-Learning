import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";
import "./DataScientistDashboard.css";
import socket from "../socket";

export default function DataScientistDashboard() {
  const navigate = useNavigate();

  const [envoys, setEnvoys] = useState([]);
  const [newEnvoy, setNewEnvoy] = useState({ name: "", s3_bucket: "", s3_prefix: "", region: "eu-north-1" });
  const [selectedEnvoyIds, setSelectedEnvoyIds] = useState([]);
  const [globalInferenceResult, setGlobalInferenceResult] = useState(null);
  const [wsMessages, setWsMessages] = useState([]);
  const [loadingStates, setLoadingStates] = useState({});
  const [selectedDeleteId, setSelectedDeleteId] = useState("");

  useEffect(() => {
    fetchEnvoys();
  }, []);

  useEffect(() => {
    socket.onopen = () => {
      console.log("✅ WebSocket connected in DataScientistDashboard");
      socket.send("Data Scientist dashboard loaded");
    };

    socket.onmessage = (event) => {
      console.log("📨 WebSocket message:", event.data);
      setWsMessages((prev) => [...prev, event.data]);
    };

    return () => {
      socket.onmessage = null;
    };
  }, []);

  const fetchEnvoys = async () => {
    try {
      const res = await api.get("/list_envoys");
      setEnvoys(res.data);
    } catch (err) {
      alert("Failed to fetch hospitals.");
      console.error(err);
    }
  };

  const createEnvoy = async (e) => {
    e.preventDefault();
    try {
      await api.post("/create_envoy", newEnvoy); // ✅ send JSON body instead of query params
      alert("✅ Hospital created!");
      setNewEnvoy({ name: "", s3_bucket: "", s3_prefix: "", region: "eu-north-1" });
      fetchEnvoys();
    } catch (err) {
      alert(`❌ Failed to create hospital: ${err.response?.data?.detail || "Unknown error"}`);
      console.error(err);
    }
  };

  const runAction = async (endpoint, message, envoyId = null) => {
    try {
      setLoadingStates((prev) => ({ ...prev, [message]: true }));

      let finalUrl = endpoint;
      let res;

      if (endpoint === "/aggregate") {
        if (selectedEnvoyIds.length === 0) {
          alert("Please select at least one hospital for aggregation.");
          return;
        }
        res = await api.post(endpoint, selectedEnvoyIds, {
          headers: { "Content-Type": "application/json" },
        });
      } else if (endpoint.includes("{envoy_id}")) {
        if (!envoyId) {
          alert("Please select a hospital.");
          return;
        }
        finalUrl = endpoint.replace("{envoy_id}", envoyId);
        if (endpoint.includes("/inference/global")) {
          res = await api.get(finalUrl);
          setGlobalInferenceResult(res.data);
        } else {
          res = await api.post(finalUrl);
        }
      } else {
        res = await api.post(endpoint);
      }

      const msg = `✅ ${message} complete.`;
      if (res.data?.s3_uri) {
        alert(`${msg}\nOutput S3 Path:\n${res.data.s3_uri}`);
      } else {
        alert(msg);
      }
    } catch (err) {
      const errorDetail = err.response?.data?.detail || err.message;
      alert(`❌ ${message} failed: ${errorDetail}`);
      console.error(err);
    } finally {
      setLoadingStates((prev) => ({ ...prev, [message]: false }));
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/login");
  };

  const handleDeleteEnvoy = async () => {
    if (!selectedDeleteId) {
      alert("Please select a hospital to delete.");
      return;
    }
    const confirm = window.confirm("Are you sure you want to delete this hospital?");
    if (!confirm) return;

    try {
      await api.delete(`/delete_envoy/${selectedDeleteId}`);
      alert("✅ Hospital deleted.");
      setSelectedDeleteId("");
      fetchEnvoys();
    } catch (err) {
      alert(`❌ Delete failed: ${err.response?.data?.detail || err.message}`);
    }
  };

  return (
    <div className="dashboard">
      <h2>Data Scientist Dashboard</h2>
      <button className="logout-button" onClick={handleLogout}>Logout</button>

      <div className="card">
        <h3>Create Hospital</h3>
        <form onSubmit={createEnvoy}>
          <input placeholder="Hospital Name" value={newEnvoy.name} onChange={(e) => setNewEnvoy({ ...newEnvoy, name: e.target.value })} required />
          <input placeholder="S3 Bucket" value={newEnvoy.s3_bucket} onChange={(e) => setNewEnvoy({ ...newEnvoy, s3_bucket: e.target.value })} required />
          <input placeholder="S3 Prefix" value={newEnvoy.s3_prefix} onChange={(e) => setNewEnvoy({ ...newEnvoy, s3_prefix: e.target.value })} required />
          <input placeholder="Region" value={newEnvoy.region} onChange={(e) => setNewEnvoy({ ...newEnvoy, region: e.target.value })} required />
          <button type="submit">Create</button>
        </form>
      </div>

      <div className="card notice-card">
        <h3>⚠️ Important Notice</h3>
        <p>
          Liaise with the resident doctor before any training. Share the training materials only first, and wait for the hospital to treat a number of patients.
          Only then should you launch training.
        </p>
        <p>Run Global Inference only after training MLs on number of envoys successfully and the doctor exporting the materials to thier individual S3 buckets, to avoid runtime errors</p>

        <p>Reach out to the lead Data Scientist Benard via benard@techlife.africa for any assistance.</p>
      </div>

      <div className="card">
        <h3>Hospital Actions</h3>
        <label>
          Select Hospital:
          <select onChange={(e) => setSelectedEnvoyIds([parseInt(e.target.value)])} value={selectedEnvoyIds[0] || ""}>
            <option value="">-- Choose --</option>
            {envoys.map((envoy) => (
              <option key={envoy.id} value={envoy.id}>{envoy.name}</option>
            ))}
          </select>
        </label>
        <div className="button-group">
          <button
            onClick={() => runAction("/package-training-code/{envoy_id}", "Packaging Code", selectedEnvoyIds[0])}
            disabled={loadingStates["Packaging Code"]}
          >
            {loadingStates["Packaging Code"] ? (<span><span className="training-bar" /> Packaging...</span>) : "Package Training Code"}
          </button>

          <button
            onClick={() => runAction("/envoy/{envoy_id}/launch_training", "Launching Training", selectedEnvoyIds[0])}
            disabled={loadingStates["Launching Training"]}
          >
            {loadingStates["Launching Training"] ? (<span><span className="training-bar" /> Training...</span>) : "Launch Training"}
          </button>

          <button
            onClick={() => runAction("/copy-trained-artifacts/{envoy_id}", "Copying Artifacts", selectedEnvoyIds[0])}
            disabled={loadingStates["Copying Artifacts"]}
          >
            {loadingStates["Copying Artifacts"] ? (<span><span className="training-bar" /> Copying...</span>) : "Copy Artifacts"}
          </button>

          <button
            onClick={() => runAction("/envoy/{envoy_id}/inference/global", "Running Inference", selectedEnvoyIds[0])}
            disabled={loadingStates["Running Inference"]}
          >
            {loadingStates["Running Inference"] ? (<span><span className="training-bar" /> Inferring...</span>) : "Run Global Inference"}
          </button>
        </div>

        {globalInferenceResult && (
          <div className="result">
            <h4>Inference Results</h4>
            <pre>{JSON.stringify(globalInferenceResult, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="card">
        <h3>Federated Learning Ops</h3>
        <div className="checkbox-group">
          <p>Select hospitals to aggregate:</p>
          <div className="checkbox-scroll">
            {envoys.map((envoy) => (
              <label key={envoy.id}>
                <input
                  type="checkbox"
                  checked={selectedEnvoyIds.includes(envoy.id)}
                  onChange={(e) => {
                    const updated = e.target.checked
                      ? [...selectedEnvoyIds, envoy.id]
                      : selectedEnvoyIds.filter((id) => id !== envoy.id);
                    setSelectedEnvoyIds(updated);
                  }}
                />
                {envoy.name}
              </label>
            ))}
          </div>
        </div>
        <div className="button-group">
          <button
            onClick={() => runAction("/aggregate", "Aggregation")}
            disabled={loadingStates["Aggregation"]}
          >
            {loadingStates["Aggregation"] ? (<span><span className="training-bar" /> Aggregating...</span>) : "Aggregate Models"}
          </button>

          <button
            onClick={() => runAction("/distribute-global-model", "Distribution")}
            disabled={loadingStates["Distribution"]}
          >
            {loadingStates["Distribution"] ? (<span><span className="training-bar" /> Distributing...</span>) : "Distribute Global Model"}
          </button>
        </div>
      </div>

      <div className="card">
        <h3>All Hospitals</h3>
        <table className="envoy-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>S3 Bucket</th>
              <th>S3 Prefix</th>
              <th>Region</th>
            </tr>
          </thead>
          <tbody>
            {envoys.map((envoy) => (
              <tr key={envoy.id}>
                <td>{envoy.id}</td>
                <td>{envoy.name}</td>
                <td>{envoy.s3_bucket}</td>
                <td>{envoy.s3_prefix}</td>
                <td>{envoy.region}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Delete Hospital</h3>
        <select value={selectedDeleteId} onChange={(e) => setSelectedDeleteId(e.target.value)}>
          <option value="">-- Select Hospital to Delete --</option>
          {envoys.map((envoy) => (
            <option key={envoy.id} value={envoy.id}>{envoy.name}</option>
          ))}
        </select>
        <button onClick={handleDeleteEnvoy} className="delete-button">
          Delete Selected Hospital
        </button>
      </div>

      {wsMessages.length > 0 && (
        <div className="card">
          <h3>Live Server Messages</h3>
          <ul className="ws-log">
            {wsMessages.map((msg, idx) => (
              <li key={idx}>{msg}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
