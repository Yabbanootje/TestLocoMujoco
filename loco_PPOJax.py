import os
import sys
import jax
import jax.numpy as jnp
from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
import gymnasium as gym

import yaml
from omegaconf import DictConfig, OmegaConf


os.environ['XLA_FLAGS'] = (
            '--xla_gpu_triton_gemm_any=True ')

training_steps = 10_000_000
algorithm = "PPO"
env_id = "MjxSkeletonMuscle"
model_name = algorithm + "_" + env_id + "_" + str(training_steps)
dir = f"./{model_name}/"
if not os.path.exists(model_name):
   os.makedirs(model_name)

yaml_path = "./conf.yaml"
with open(yaml_path, "r") as file:
    try:
        config = DictConfig(yaml.load(file, Loader=yaml.FullLoader))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'{yaml_path} error: {exc}') from exc

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

# Initialize the environments
# note: we do not support parallel environments in gymnasium yet!
# env = gym.make("LocoMujoco", env_name="SkeletonMuscle", render_mode="rgb_array",
#             #    default_dataset_conf=DefaultDatasetConf("walk"),
#             #    lafan1_dataset_conf=LAFAN1DatasetConf("walk1_subject1"),
#             #    goal_type="GoalTrajMimicv2", goal_params=dict(visualize_goal=True),
#                 record = False,
#                 recorder_params = {"path": dir+"videos/", "tag": model_name}
#                )

# get initial agent configuration
agent_conf = PPOJax.init_agent_conf(env, config)

# build training function
train_fn = PPOJax.build_train_fn(env, agent_conf)

# jit and vmap training function
train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

# get rng keys and run training
rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
print("Starting training...")
out = train_fn(_rng)
print("Training completed.")

# save agent state
agent_state = out["agent_state"]
save_path = PPOJax.save_agent(dir, agent_conf, agent_state)


# load agent
agent_conf, agent_state = PPOJax.load_agent(dir + "PPOJax_saved.pkl")
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

import time
t_start = time.time()
print("Logging metrics...")
# get the metrics and log them
if not config.experiment.debug:
    training_metrics = out["training_metrics"]
    validation_metrics = out["validation_metrics"]

    # calculate mean across seeds
    training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
    validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

    for i in range(len(training_metrics.mean_episode_return)):
        print({"Mean Episode Return": training_metrics.mean_episode_return[i],
                    "Mean Episode Length": training_metrics.mean_episode_length[i]},
                f"step={int(training_metrics.max_timestep[i])}")

        if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
            print({"Validation Info/Mean Episode Return": validation_metrics.mean_episode_return[i],
                        "Validation Info/Mean Episode Length": validation_metrics.mean_episode_length[i]},
                    f"step={int(training_metrics.max_timestep[i])}")

            # log all measures
            metrics_to_log = {}
            for field in fields(validation_metrics):
                attr = getattr(validation_metrics, field.name)
                if isinstance(attr, QuantityContainer):
                    measure_name = field.name
                    for field_attr in fields(attr):
                        attr_name = field_attr.name
                        attr_value = getattr(attr, attr_name)
                        if attr_value.size > 0:
                            metrics_to_log[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

            # metric for used for wandb sweep (optional)
            site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
            site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
            site_rvel = validation_metrics.euclidean_distance.site_rpos[i]

print(f"Time taken to log metrics: {time.time() - t_start}s")

# run the environment with the trained agent to record video
PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=1, record=True,
                    train_state_seed=0)
video_file = env.video_file_path