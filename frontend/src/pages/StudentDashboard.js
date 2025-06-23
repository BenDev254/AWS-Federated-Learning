// components/StudentDashboard.js
import React from "react";
// import { getCurrentUser, logout } from "../auth";
export default function StudentDashboard() {
//   const u = getCurrentUser();
  return (
    <div>
      <h2>Student Dashboard</h2>
      <p>Welcome, </p>
      {/* <p>Welcome, {u.username}</p>
      <button onClick={() => { logout(); window.location="/login"; }}>Logout</button> */}
    </div>
  );
}
