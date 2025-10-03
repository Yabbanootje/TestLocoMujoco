from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
# from loco_mujoco.utils.video_recorder import VideoRecorder

def render_video(model, env, video_folder="logs/videos/", video_length=100, name_prefix="trained-agent"):
    """
    Render a video of a trained model in a given environment.

    Args:
        model: The trained RL model.
        env (str): The environment.
        video_folder (str): The folder to save the video.
        video_length (int): The length of the video in steps.
        name_prefix (str): The prefix for the video file name.
    """
    
    vec_env = DummyVecEnv([lambda: env])

    obs = vec_env.reset()

    # Record the video starting at the first step
    vec_env = VecVideoRecorder(vec_env, video_folder,
                           record_video_trigger=lambda x: x == 0, video_length=video_length,
                           name_prefix=name_prefix)

    vec_env.reset()
    for _ in range(video_length + 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _ = vec_env.step(action)
    # Save the video
    vec_env.close()

    # recorder = VideoRecorder(video_folder)