import time
import gymnasium as gym

# Creates Mujoco Inverted pendulum and shows the image.
env = gym.make("InvertedPendulum-v5", render_mode="human")

try:
    # 5 turns
    for episode in range(5):
        observation, info = env.reset()
        step_count = 0

        print(f"\nEpisode {episode + 1}")
        print("Initial observation:", observation)

        while True:
            # choose a random thrust
            action = env.action_space.sample()

            # take action and ge the result
            observation, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # turn into 4 components
            x, angle, velocity, angular_velocity = observation

            print(
                f"Step {step_count}: "
                f"action={action[0]:.2f}, "
                f"x={x:.3f}, "
                f"angle={angle:.3f}, "
                f"velocity={velocity:.3f}, "
                f"angular_velocity={angular_velocity:.3f}, "
                f"reward={reward}"
            )

            # slow down for observation
            time.sleep(0.5)

            if terminated or truncated:
                print(
                    f"Episode ended after {step_count} steps. "
                    f"terminated={terminated}, truncated={truncated}"
                )
                break

finally:
    env.close()
