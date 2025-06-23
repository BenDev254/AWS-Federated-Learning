import os
import torch
import torch.nn as nn


# Define model architecture (must match exactly how you trained)
class RiskModel(nn.Module):
    def __init__(self):
        super(RiskModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.seq(x)

    def state_dict(self, *args, **kwargs):
        return self.seq.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.seq.load_state_dict(state_dict, *args, **kwargs)


# List envoy folders
received_dir = "./received_results"
envoy_dirs = [os.path.join(received_dir, d) for d in os.listdir(received_dir)
              if os.path.isdir(os.path.join(received_dir, d))]

# Load all state_dicts
state_dicts = []
for envoy_path in envoy_dirs:
    model_path = os.path.join(envoy_path, "model.pt")
    if os.path.exists(model_path):
        print(f"Loading: {model_path}")
        state_dict = torch.load(model_path)
        state_dicts.append(state_dict)
    else:
        print(f"Model not found: {model_path}")

# Ensure at least two models
if len(state_dicts) < 2:
    raise ValueError("Need at least two models for aggregation.")

# Initialize averaged model
avg_model = RiskModel()
avg_state_dict = avg_model.state_dict()

# Initialize parameter sums
for key in avg_state_dict.keys():
    avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

# Load averaged weights
avg_model.load_state_dict(avg_state_dict)

# Save aggregated model
os.makedirs("./aggregated_model", exist_ok=True)
torch.save(avg_model.state_dict(), "./aggregated_model/model.pt")

print("✅ Aggregated model saved to ./aggregated_model/model.pt")
