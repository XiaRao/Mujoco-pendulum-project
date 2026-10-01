"""file"""
import time
import gymnasium as gym
import numpy as np
from pathlib import Path

# Creates Mujoco Inverted pendulum and shows the image.
# render_mode = "human" means demonstrating visible page, the fundamental movements
# are simulated by mujoco.
env = gym.make("InvertedPendulum-v5")

try:
    # 5 episodes
    transitions = []
    episode = 0
    while len(transitions) < 100:

        # Starting a new attempt.
        # observation refers to the initial observation, info is extra information.
        observation, info = env.reset()
        step_count = 0

        print(f"\nEpisode {episode + 1}")
        print("Initial observation:", observation)

        while len(transitions) < 100:
            # choose a random thrust. 1 digit
            action = env.action_space.sample()

            # stores the current observation for collection. 4 digit
            state = observation.copy()

            # env.step(action) executes a thrust, returns the result of that.
            # each represents 1 digit.
            observation, reward, terminated, truncated, info = env.step(action)

            transitions.append({
                "state": state,
                "action": action.copy(),
                "next_state": observation.copy(),
                "episode_id": episode,
                "terminated": terminated,
                "truncated": truncated,
            })

            step_count += 1

            # turn into 4 components
            x, angle, velocity, angular_velocity = observation

            # collect the data.
            print(
                f"Step {step_count}: "
                f"action={action[0]:.2f}, "
                f"x={x:.3f}, "
                f"angle={angle:.3f}, "
                f"velocity={velocity:.3f}, "
                f"angular_velocity={angular_velocity:.3f}, "
                f"reward={reward}"
            )

            # The gymnasium environment has pre-set the termination and truncation.
            # abs(observation[1]) > 0.2 or max 1000 steps.
            if terminated or truncated:
                print(
                    f"Episode ended after {step_count} steps. "
                    f"terminated={terminated}, truncated={truncated}"
                )
                episode += 1
                break

    print("Number of transitions: ", len(transitions))
    print("First Transition: ", transitions[0])
    print("Last Transition: ", transitions[-1])

    # transition is currently a dictionary.
    # Convert the collected records into stacks
    states = np.stack([item["state"] for item in transitions])
    actions = np.stack([item["action"] for item in transitions])
    next_states = np.stack([item["next_state"] for item in transitions])

    # array is used because these variables contain one int only.
    episode_ids = np.array([item["episode_id"] for item in transitions])
    terminated_flags = np.array([item["terminated"] for item in transitions])
    truncated_flags = np.array([item["truncated"] for item in transitions])

    # Save the dataset next to this script.
    output_path = Path(__file__).resolve().parent / "pendulum_data.npz"

    # compress the 6 arrays into a file
    np.savez_compressed(
        output_path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        terminated_flags=terminated_flags,
        truncated_flags=truncated_flags,
    )

    print("Saved dataset to: ", output_path)

    # Reload the dataset and check its shapes and values.
    with np.load(output_path) as data:
        print("States shape: ", data["states"].shape)
        print("Actions shape: ", data["actions"].shape)
        print("Next States shape: ", data["next_states"].shape)

        assert np.array_equal(data["states"], states)
        assert np.array_equal(data["actions"], actions)
        assert np.array_equal(data["next_states"], next_states)

finally:
    env.close()
