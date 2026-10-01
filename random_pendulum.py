"""file"""
import time
import gymnasium as gym

# Creates Mujoco Inverted pendulum and shows the image.
# render_mode = "human" means demonstrating visible page, the fundamental movements
# are simulated by mujoco.
env = gym.make("InvertedPendulum-v5", render_mode="human")

try:
    # 5 episodes
    for episode in range(5):

        # Starting a new attempt.
        # observation refers to the initial observation, info is extra information.
        observation, info = env.reset()
        step_count = 0

        print(f"\nEpisode {episode + 1}")
        print("Initial observation:", observation)

        while True:
            # choose a random thrust
            action = env.action_space.sample()

            # env.step(action) executes a thrust, returns the result of that.
            observation, reward, terminated, truncated, info = env.step(action)
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

            # slow the process down for observation
            time.sleep(0.5)

            # The gymnasium environment has pre-set the termination and truncation.
            # abs(observation[1]) > 0.2 or max 1000 steps.
            if terminated or truncated:
                print(
                    f"Episode ended after {step_count} steps. "
                    f"terminated={terminated}, truncated={truncated}"
                )
                break

finally:
    env.close()
