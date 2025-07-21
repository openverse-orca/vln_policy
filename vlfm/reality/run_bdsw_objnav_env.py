# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from spot_wrapper.spot import Spot

import os,sys
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(project_root)
print(current_file_path, "---",project_root)
sys.path.append(project_root)
os.chdir(project_root)

# from vlfm.policy.reality_policies import RealityConfig, RealityITMPolicyV2
# from vlfm.reality.objectnav_env import ObjectNavEnv
# from vlfm.reality.robots.bdsw_robot import BDSWRobot

from  vlfm.policy.reality_policies import RealityConfig, RealityITMPolicyV2
from  vlfm.reality.objectnav_env import ObjectNavEnv
from  vlfm.reality.robots.bdsw_robot import BDSWRobot

# 如果timm报错， 可尝试pip install timm==0.6.12
# https://github.com/Extraltodeus/depthmap2mask/issues/35
@hydra.main(version_base=None, config_path=project_root+"/config", config_name="experiments/reality")
def run_policy_orca(cfg: RealityConfig) -> None:
    print("2222222222222222")
    print(OmegaConf.to_yaml(cfg))
    policy = RealityITMPolicyV2.from_config(cfg)

    print("1111111111111111")
    spot = Spot("BDSW_env")  # just a name, can be anything
    print("OKOKOKOKOKOKOKOK")
    with spot.get_lease(True):  # turns the robot on, and off upon any errors/completion
        spot.power_on()
        spot.blocking_stand()
        robot = BDSWRobot(spot)
        robot.open_gripper()
        # robot.set_arm_joints(NOMINAL_ARM_POSE, travel_time=0.75)
        cmd_id = robot.spot.move_gripper_to_point(np.array([0.35, 0.0, 0.6]), np.deg2rad([0.0, 20.0, 0.0]))
        spot.block_until_arm_arrives(cmd_id, timeout_sec=1.5)
        env = ObjectNavEnv(
            robot=robot,
            max_body_cam_depth=cfg.env.max_body_cam_depth,
            max_gripper_cam_depth=cfg.env.max_gripper_cam_depth,
            max_lin_dist=cfg.env.max_lin_dist,
            max_ang_dist=cfg.env.max_ang_dist,
            time_step=cfg.env.time_step,
        )
        goal = cfg.env.goal
        run_env(env, policy, goal)


def run_env(env: ObjectNavEnv, policy: RealityITMPolicyV2, goal: str) -> None:
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device="cuda", dtype=torch.bool)
    st = time.time()
    action = policy.get_action(observations, mask)
    print(f"get_action took {time.time() - st:.2f} seconds")
    while not done:
        observations, _, done, info = env.step(action)
        st = time.time()
        action = policy.get_action(observations, mask, deterministic=True)
        print(f"get_action took {time.time() - st:.2f} seconds")
        mask = torch.ones_like(mask)
        if done:
            print("Episode finished because done is True")
            break


if __name__ == "__main__":
    run_policy_orca()
