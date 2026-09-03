import os
import sys
from ale_py import env
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import wandb

from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf
from loco_mujoco.algorithms import PPOJax, PPOAgentConf, PPOAgentState
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

from loco_mujoco.smpl.const import KINESIS_TRAIN_LOCOMOTION_DATASETS, KINESIS_TEST_LOCOMOTION_DATASETS

SERVER = False
TRAIN = False
VALIDATE = False
PLAY = True
PLOT = True


@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:
        if not SERVER:
            training_steps = 50000
            algorithm = "PPO"
            env_id = "MjxSkeletonTorque"
            model_name = algorithm + "_" + env_id + "_" + str(training_steps) #+ "_CVAE_64_HighjumpSquatRunWalk"
            # model_name = "PPO_MjxSkeletonMuscle_500000000"
            model_name = "PPO_MjxSkeletonTorque_300000000_Future/baseline-0"
            # model_name = "PPO_MjxSkeletonTorque_75000000/one_walk"
            dir = f"./{model_name}/"
            if not os.path.exists(model_name):
                os.makedirs(model_name)

        if SERVER and TRAIN:
            os.environ['XLA_FLAGS'] = (
                '--xla_gpu_triton_gemm_any=True ')

            # Accessing the current sweep number
            result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

            # # setup wandb
            # wandb.login()
            # config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
            # run = wandb.init(project=config.wandb.project, config=config_dict)

            # get task factory
            factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

            # create env
            env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

            # get initial agent configuration
            agent_conf = PPOJax.init_agent_conf(env, config)

            # setup metric handler (optional)
            mh = MetricsHandler(config, env) if config.experiment.validation.active else None

            # build training function
            train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

            # jit and vmap training function
            train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

            # get rng keys and run training
            rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
            rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
            out = train_fn(_rng)

            # save agent state
            agent_state = out["agent_state"]
            save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
            # run.config.update({"agent_save_path": save_path})

            import time
            t_start = time.time()
            # get the metrics and log them
            if not config.experiment.debug:
                training_metrics = out["training_metrics"]
                validation_metrics = out["validation_metrics"]

                # calculate mean across seeds
                training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), training_metrics)
                validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), validation_metrics)

                for i in range(len(training_metrics.mean_episode_return)):
                    print({"Mean Episode Return": training_metrics.mean_episode_return[i],
                        "Var Episode Return": training_metrics.var_episode_return[i],
                        "Mean Episode Length": training_metrics.mean_episode_length[i],
                        "Var Episode Length": training_metrics.var_episode_length[i],
                        "Num Episodes": training_metrics.num_episodes[i],
                        "Max Episode Return": training_metrics.max_episode_return[i],
                        "Min Episode Return": training_metrics.min_episode_return[i],
                        "Max Episode Length": training_metrics.max_episode_length[i],
                        "Min Episode Length": training_metrics.min_episode_length[i],
                        "Max Timestep": training_metrics.max_timestep[i],
                        "Min Timestep": training_metrics.min_timestep[i],
                        },
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

                        # run.log(metrics_to_log, step=int(training_metrics.max_timestep[i]))

                        # metric for used for wandb sweep (optional)
                        site_rpos = validation_metrics.euclidean_distance.site_rpos[i]
                        site_rrotvec = validation_metrics.euclidean_distance.site_rpos[i]
                        site_rvel = validation_metrics.euclidean_distance.site_rpos[i]
                        # run.log({"Metric for Sweep": site_rpos + site_rrotvec + site_rvel},
                        #         step=int(training_metrics.max_timestep[i]))

            print(f"Time taken to log metrics: {time.time() - t_start}s")

        if VALIDATE:
            # load agent
            agent_conf, agent_state = PPOJax.load_agent(dir + "PPOJax_saved.pkl")
            config = agent_conf.config

            if hasattr(config.experiment, "cvae") and config.experiment.cvae.use_cvae:
                cvae_agent_conf, cvae_agent_state = PPOJax.load_agent(dir + "cvae/" + "PPOJax_saved.pkl")
                agent_conf = PPOAgentConf(
                    config=agent_conf.config,
                    network=(agent_conf.network, cvae_agent_conf.network),
                    tx=agent_conf.tx,
                )
                agent_state = PPOAgentState(train_state=(agent_state.train_state, cvae_agent_state.train_state))

            # validate CVAE

            # config["recorder_params"] = {"path": dir, "tag": "videos/"}
            
            if SERVER:
                # create env
                OmegaConf.set_struct(config, False)  # Allow modifications
                config.experiment.env_params["th_params"] = {"random_start": False, "fixed_start_conf": [0, 0]}
                config.experiment.task_factory.params.amass_dataset_conf["dataset_group"] = "KINESIS_TRAIN_LOCOMOTION_DATASETS"
                config.experiment.env_params["horizon"] = 100000

                # get task factory
                factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
                env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

                PPOJax.play_all_trajectories(env, agent_conf, agent_state, deterministic=True, n_steps=None, n_envs=1, 
                                             render=False, record=False, train_state_seed=0, 
                                             save_metrics_folder=dir+"train-metrics/", use_mujoco=False, 
                                             trajectory_names=KINESIS_TRAIN_LOCOMOTION_DATASETS)
            
            # create env
            OmegaConf.set_struct(config, False)  # Allow modifications
            config.experiment.env_params["th_params"] = {"random_start": False, "fixed_start_conf": [0, 0]}
            config.experiment.task_factory.params.amass_dataset_conf["dataset_group"] = "KINESIS_TEST_LOCOMOTION_DATASETS"
            config.experiment.env_params["horizon"] = 100000

            # get task factory
            factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
            env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

            PPOJax.play_all_trajectories(env, agent_conf, agent_state, deterministic=True, n_steps=None, n_envs=1, 
                                         render=True, record=False, train_state_seed=0,
                                         save_metrics_folder=dir+"test-metrics/", use_mujoco=False,
                                         trajectory_names=KINESIS_TEST_LOCOMOTION_DATASETS)

        if not SERVER and PLAY:
            # env = ImitationFactory.make("SkeletonTorque",
            #                 default_dataset_conf=DefaultDatasetConf(["walk"]),
            #                 # lafan1_dataset_conf=LAFAN1DatasetConf(["dance2_subject4", "walk1_subject1"]),
            #                 # if SMPL and AMASS are installed, you can use the following:
            #                 # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
            #                 #                                     "KIT/12/WalkInClockwiseCircle11_poses",
            #                 #                                     "HUMAN4D/HUMAN4D/Subject3_Medhi/INF_JumpingJack_S3_01_poses",
            #                 #                                     'KIT/359/walking_fast05_poses']),
            #                 # n_substeps=20
            #                 th_params=dict(random_start=False, fixed_start_conf=[0, 0])
            #                 )

            # env.play_trajectory(n_episodes=1, record=True, render=True)
            # video_file = env.video_file_path
            # print({"Save video to": video_file})

            # load agent
            agent_conf, agent_state = PPOJax.load_agent(dir + "PPOJax_saved.pkl")
            config = agent_conf.config

            if hasattr(config.experiment, "cvae") and config.experiment.cvae.use_cvae:
                cvae_agent_conf, cvae_agent_state = PPOJax.load_agent(dir + "cvae/" + "PPOJax_saved.pkl")
                agent_conf = PPOAgentConf(
                    config=agent_conf.config,
                    network=(agent_conf.network, cvae_agent_conf.network),
                    tx=agent_conf.tx,
                )
                agent_state = PPOAgentState(train_state=(agent_state.train_state, cvae_agent_state.train_state))

            config["recorder_params"] = {"path": dir, "tag": "videos/"}

            # get task factory
            factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

            # create env
            OmegaConf.set_struct(config, False)  # Allow modifications
            config.experiment.env_params["headless"] = False
            config.experiment.env_params["horizon"] = 2000
            # config.experiment.env_params.th_params["random_start"] = False
            # config.experiment.env_params.th_params["fixed_start_conf"] = [0, 0]
            config.experiment.env_params["th_params"] = {"random_start": True} #False, "fixed_start_conf": [0, 0]}
            config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"
            # config.experiment.env_params.pop("scaling")
            # config.experiment.env_params.goal_params["n_step_lookahead"] = 0
            # config.experiment.env_params.pop("use_box_feet")
            # config.experiment.env_params.goal_params["visualize_goal"] = False
            # config.experiment.task_factory.params.amass_dataset_conf["dataset_group"] = "KINESIS_TEST_LOCOMOTION_DATASETS"

            # for task in config.experiment.task_factory.params.default_dataset_conf["task"]:
            # config.experiment.task_factory.params.amass_dataset_conf["dataset_group"] = None
            # config.experiment.task_factory.params.amass_dataset_conf["rel_dataset_path"] = [
                                                                    #  "KIT/12/WalkingStraightForwards01_1_poses.npz", 
                                                                    #  "KIT/12/WalkingStraightForwards03_poses.npz", 
                                                                    #  "KIT/12/WalkingStraightForwards04_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards05_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards06_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards07_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards08_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards09_1_poses.npz",
                                                                    #  "KIT/12/WalkingStraightForwards10_poses.npz",
                                                                    #  "KIT/11/WalkingStraightForwards05_poses.npz",
                                                                    #  "KIT/12/LeftTurn03_poses",
                                                                    #  "KIT/6/WalkingStraightBackwards04_poses.npz",
                                                                    #  "KIT/7/WalkingStraightBackwards04_poses.npz",
                                                                    #  ]

            # env = ImitationFactory.make("MjxSkeletonTorque",
            #                     # if SMPL and AMASS are installed, you can use the following:
            #                     amass_dataset_conf=AMASSDatasetConf([
            #                                                         #  "KIT/12/WalkingStraightForwards01_1_poses.npz", 
            #                                                          "KIT/12/WalkingStraightForwards03_poses.npz", 
            #                                                         #  "KIT/205/walking_medium01_poses.npz",
            #                                                         #  "KIT/167/walking_medium03_poses",
            #                                                         #  "KIT/12/WalkingStraightForwards04_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards05_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards06_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards07_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards08_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards09_1_poses.npz",
            #                                                         #  "KIT/12/WalkingStraightForwards10_poses.npz"
            #                                                          ]),
            #                     **{"goal_type": "GoalTrajMimicv2"}
            #                     # n_substeps=20
            #                     )

            # env.play_trajectory(n_episodes=1, n_steps_per_episode=5000, render=True)

            env = factory.make(**config.experiment.env_params, **config.experiment.task_factory.params)

            # run the environment with the trained agent to record video
            PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=2000, n_envs=1, record=True,
                            train_state_seed=0, save_kinematics=True, save_kinematics_folder=dir, use_mujoco=False)
            video_file = env.video_file_path
            print({"Save video to": video_file})

        # wandb.finish()

        if not SERVER and PLOT:
            # Load the two files
            actuators = "torques"
            motion = ""#"jumping_"
            agent_kinematics_df = pd.read_csv(dir + motion + "kinematics/agent_motion_kinematics.csv")#.iloc[:500]
            reference_kinematics_df = pd.read_csv(dir + motion + "kinematics/reference_motion_kinematics.csv")#.iloc[:500]
            agent_kinetics_df = pd.read_csv(dir + motion + f"kinematics/agent_{actuators}_kinetics.csv")#.iloc[:500]
            reference_kinetics_df = pd.read_csv(dir + motion + f"kinematics/agent_motion_joint_kinetics.csv")#.iloc[:500]
            # baseline_kinematics_df = pd.read_csv(dir + "../PPO_MjxSkeletonTorque_500000000_Baseline_HighjumpSquatRunWalk/"
            #                           + motion + "kinematics/agent_motion.csv").iloc[:500]

            angle_cols = [c for c in agent_kinematics_df.columns if c.startswith("q")]
            velocity_cols = [c for c in agent_kinematics_df.columns if c.startswith("dq")]

            rmse_per_joint = np.sqrt(np.mean(((agent_kinematics_df - reference_kinematics_df) ** 2), axis=0))
            std_rmse_q_over_joints = rmse_per_joint[angle_cols].std()
            std_rmse_dq_over_joints = rmse_per_joint[velocity_cols].std()
            rmse_q = np.sqrt(np.mean(((agent_kinematics_df[angle_cols] - reference_kinematics_df[angle_cols]) ** 2)))
            rmse_dq = np.sqrt(np.mean(((agent_kinematics_df[velocity_cols] - reference_kinematics_df[velocity_cols]) ** 2)))
            error_df = agent_kinematics_df - reference_kinematics_df
            mae_per_joint = error_df.abs().mean(axis=0)
            std_per_joint = error_df.abs().std(axis=0)
            mae_q = error_df[angle_cols].abs().to_numpy().mean()
            std_q = error_df[angle_cols].abs().to_numpy().std()
            std_q_over_joints = mae_per_joint[angle_cols].std()
            mae_dq = error_df[velocity_cols].abs().to_numpy().mean()
            std_dq = error_df[velocity_cols].abs().to_numpy().std()
            std_dq_over_joints = mae_per_joint[velocity_cols].std()

            with open(dir + motion + "kinematics/error_metrics_kinematics.txt", "w") as f:
                f.write(f"rmse_per_joint:\n{rmse_per_joint}\n")
                f.write(f"rmse_q: {rmse_q}\n")
                f.write(f"rmse_dq: {rmse_dq}\n")
                f.write(f"std_rmse_q_over_joints: {std_rmse_q_over_joints}\n")
                f.write(f"std_rmse_dq_over_joints: {std_rmse_dq_over_joints}\n")
                f.write(f"mae_per_joint:\n{mae_per_joint}\n")
                f.write(f"std_per_joint:\n{std_per_joint}\n")
                f.write(f"mae_q: {mae_q}\n")
                f.write(f"std_q: {std_q}\n")
                f.write(f"std_q_over_joints: {std_q_over_joints}\n")
                f.write(f"mae_dq: {mae_dq}\n")
                f.write(f"std_dq: {std_dq}\n")
                f.write(f"std_dq_over_joints: {std_dq_over_joints}\n")

            # rmse_per_actuator = np.sqrt(np.mean(((agent_kinetics_df) - reference_kinetics_df)) ** 2), axis=0))
            # std_rsme_over_actuators = rmse_per_actuator.std()
            # rmse = np.sqrt(np.mean(((agent_kinetics_df) - reference_kinetics_df)) ** 2)))
            # error_df = agent_kinetics_df - reference_kinetics_df)
            # mae_per_actuator = error_df.abs().mean(axis=0)
            # std_per_actuator = error_df.abs().std(axis=0)
            # mae = error_df.abs().to_numpy().mean()
            # std = error_df.abs().to_numpy().std()
            # std_over_actuators = mae_per_actuator.std()

            # with open(dir + motion + "kinematics/error_metrics_kinetics.txt", "w") as f:
            #     f.write(f"rmse_per_actuator: {rmse_per_actuator}\n")
            #     f.write(f"rmse: {rmse}\n")
            #     f.write(f"std_rsme_over_actuators: {std_rsme_over_actuators}\n")
            #     f.write(f"mae_per_actuator: {mae_per_actuator}\n")
            #     f.write(f"std_per_actuator: {std_per_actuator}\n")
            #     f.write(f"mae: {mae}\n")
            #     f.write(f"std: {std}\n")
            #     f.write(f"std_over_actuators: {std_over_actuators}\n")

            # joints = ["q_knee_angle_l", "q_knee_angle_r"]  # joint to compare
            # joint = "q_knee_angle_r"
            joints = [
                ("q_knee_angle_l", "q_knee_angle_r"),
                ("dq_knee_angle_l", "dq_knee_angle_r"),
                # ("q_hip_rotation_l", "q_hip_rotation_r"),
                # ("q_pro_sup_l", "q_pro_sup_r"),
                # ("q_ankle_angle_l", "q_ankle_angle_r"),
            ]
            
            for left_joint, right_joint in joints:
                if 'd' in left_joint.split('_')[0]:
                    metric = "velocity"
                else:
                    metric = "angle"

                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

                ax1.plot(reference_kinematics_df["Timestep"]/100, reference_kinematics_df[left_joint], label="Reference")
                ax1.plot(agent_kinematics_df["Timestep"]/100, agent_kinematics_df[left_joint], label="CVAE", alpha=0.5)#, color='r')
                ax1.set_title(f"Left {left_joint.split('_')[1]} {metric}")
                ax1.set_xlabel("Time (sec)")
                ax1.set_ylabel(f"Joint {metric} (degrees{'/sec' if metric == 'velocity' else ''})")
                # ax1.set_ylim(-1, 72)
                ax1.legend(loc="lower left")
                ax1.grid(True)

                ax2.plot(reference_kinematics_df["Timestep"]/100, reference_kinematics_df[right_joint], label="Reference")
                # ax2.plot(baseline_kinematics_df["Timestep"]/100, baseline_kinematics_df[right_joint], label="Baseline", alpha=0.5)#, color='y')
                ax2.plot(agent_kinematics_df["Timestep"]/100, agent_kinematics_df[right_joint], label="CVAE", alpha=0.5)#, color='r')
                ax2.set_title(f"Right {right_joint.split('_')[1]} {metric}")
                ax2.set_xlabel("Time (sec)")
                ax2.set_ylabel(f"Joint {metric} (degrees{'/sec' if metric == 'velocity' else ''})")
                ax2.legend(loc="lower left")
                ax2.grid(True)

                plt.savefig(dir + motion + f"kinematics/{left_joint}_baseline.png", dpi=300, bbox_inches="tight") # _comparison_500
                plt.savefig(dir + motion + f"kinematics/{left_joint}_baseline.pdf", dpi=300, bbox_inches="tight", format="pdf")
                plt.close()

            actuators = [
                ("mot_knee_angle_l", "mot_knee_angle_r"),
                # ("quad_fem_l", "quad_fem_r"),
            ]
            
            for left_actuator, right_actuator in actuators:
                metric = "torque"

                fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
                fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=3.0)

                ax1.plot(reference_kinetics_df["Timestep"]/100, reference_kinetics_df[left_actuator.replace("mot_", "")], label="Reference")
                # ax1.plot(baseline_kinetics_df["Timestep"]/100, baseline_kinetics_df[left_actuator]), label="Baseline", alpha=0.5)#, color='y')
                ax1.plot(agent_kinetics_df["Timestep"]/100, agent_kinetics_df[left_actuator], label="CVAE", alpha=0.5)#, color='r')
                ax1.set_title(f"Left {left_actuator.split('_')[1]} {metric}")
                ax1.set_xlabel("Time (sec)")
                ax1.set_ylabel(f"Joint {metric} (Nm)")
                # ax1.set_ylim(-1, 72)
                ax1.legend(loc="lower left")
                ax1.grid(True)

                ax2.plot(reference_kinetics_df["Timestep"]/100, reference_kinetics_df[right_actuator.replace("mot_", "")], label="Reference")
                # ax2.plot(baseline_kinetics_df["Timestep"]/100, baseline_kinetics_df[right_actuator]), label="Baseline", alpha=0.5)#, color='y')
                ax2.plot(agent_kinetics_df["Timestep"]/100, agent_kinetics_df[right_actuator], label="CVAE", alpha=0.5)#, color='r')
                ax2.set_title(f"Right {right_actuator.split('_')[1]} {metric}")
                ax2.set_xlabel("Time (sec)")
                ax2.set_ylabel(f"Joint {metric} (Nm)")
                ax2.legend(loc="lower left")
                ax2.grid(True)

                plt.savefig(dir + motion + f"kinematics/{left_actuator}_baseline.png", dpi=300, bbox_inches="tight") # _comparison_500
                plt.savefig(dir + motion + f"kinematics/{left_actuator}_baseline.pdf", dpi=300, bbox_inches="tight", format="pdf")
                plt.close()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()