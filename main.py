import numpy as np
import loco_mujoco
from loco_mujoco.task_factories import DefaultDatasetConf, LAFAN1DatasetConf
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter

from render_video import render_video

TRAINING = True
EVALUATE = False

training_steps = 3_000
algorithm = "PPO"
env_id = "myoLegWalk-v0"
model_name = algorithm + "_" + env_id + "_" + str(training_steps)
dir = f"./{model_name}/"

# Initialize the environments
# note: we do not support parallel environments in gymnasium yet!
env = gym.make("LocoMujoco", env_name="SkeletonMuscle", render_mode="rgb_array",
            #    default_dataset_conf=DefaultDatasetConf("walk"),
            #    lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
            #    goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True),
                # record = False,
                recorder_params = {"path": dir, "tag": "videos/"}
               )
eval_env = env


if TRAINING:
    # Custom actor (pi) and value function (vf) networks
    # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[2048, 1536, 1024, 1024, 512, 512], 
                                    vf=[2048, 1536, 1024, 1024, 512, 512]))

    # Create the agent
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    # Create a callback to evaluate every 1000 steps during training
    eval_callback = EvalCallback(eval_env, best_model_save_path=dir+"eval/",
                                log_path=dir+"eval/", eval_freq=1000,
                                n_eval_episodes=5, deterministic=True,
                                render=True)
    
    # Train the agent and render the training process
    # env.mj_render()
    model.learn(total_timesteps = training_steps, callback = eval_callback)

    # Save the agent
    print(f"Saving model to {model_name}.zip")
    print(dir)
    model.save(dir + model_name)


# Load the trained agent
model = PPO.load(dir + model_name, env=eval_env)

# # Record a video of the agent
# video_folder = dir+"videos/"
# video_length = 100
# render_video(model, eval_env, video_folder=video_folder, video_length=video_length, name_prefix=model_name)


if EVALUATE:
    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), render=False, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    total_reward = 0
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        total_reward += rewards
    env.close()
    print("Total reward:", total_reward)


# Plot the results
results = np.load(dir + "eval/evaluations.npz")
print(len(results['results']), results['results'])
plt.plot(results['timesteps'], np.mean(results['results'], axis=1))
plt.show()