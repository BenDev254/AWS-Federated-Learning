const socket = new WebSocket(
  process.env.REACT_APP_WS_URL || "wss://demo-fl-backend-dheedycdf0hbbph7.canadacentral-01.azurewebsites.net/ws"
);

// Handle connection open
socket.onopen = () => {
  console.log("✅ WebSocket connection established");
};

// Handle incoming messages
socket.onmessage = (event) => {
  console.log("📨 Message received from server:", event.data);
};

export default socket;
