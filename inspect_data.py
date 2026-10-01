"""Inspects the data from collect_data"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# load the saved dataset
data_path = Path(__file__).resolve().parent / "pendulum_data.npz"

with np.load(data_path) as data:
    states = data["states"]
    next_states = data["next_states"]
    episode_ids = data["episode_ids"]

# Select the first recorded episode
selected_episode = episode_ids[0]
# Returns a boolean list. states[mask] only pick the rows that are True.
mask = episode_ids == selected_episode

episode_states = states[mask]
episode_next_states = next_states[mask]

print("Episode ID: ", selected_episode)
print("Number of transitions: ", len(episode_states))

# Each next state should match the following current state. check if the data is correct.
assert np.allclose(
    # deletes the last row
    episode_next_states[:-1],
    # deletes the first row to check if allclose.
    episode_states[1:],)

print("Trajectory continuity check passes.")

# Include the initial angle and every angle after an action.
angles = np.concatenate([
    episode_states[:1, 1],
    episode_next_states[:, 1],])

steps = np.arange(len(angles))

# creates a graph
plt.plot(steps, angles, marker="o", label="Pole angle")
plt.axhline(0.2, color="red", linestyle="--", label="Failure limits")
plt.axhline(-0.2, color="red", linestyle="--")
plt.axhline(0, color="gray", linestyle=":")

plt.xlabel("Simulation step")
plt.ylabel("Pole angle (radians)")
plt.title(f"Episode ID {selected_episode}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
