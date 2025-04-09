# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from depth_camera_filtering import filter_depth

from  vlfm.reality.pointnav_env_orca import PointNavOrcaEnv
from  vlfm.reality.robots.camera_ids import SpotCamIds
from  vlfm.utils.geometry_utils import get_fov, wrap_heading
from  vlfm.utils.img_utils import reorient_rescale_map, resize_images
from  vlfm.vlm.imgrec import RecFromRos2Img



LEFT_CROP = 124
RIGHT_CROP = 60
NOMINAL_ARM_POSE = np.deg2rad([0, -170, 120, 0, 55, 0])

VALUE_MAP_CAMS = [
    # SpotCamIds.BACK_FISHEYE,
    # SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
    # SpotCamIds.LEFT_FISHEYE,
    # SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    # SpotCamIds.RIGHT_FISHEYE,
    # SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.HAND_COLOR,
]

POINT_CLOUD_CAMS = [
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
]

ALL_CAMS = list(set(VALUE_MAP_CAMS + POINT_CLOUD_CAMS))


class ObjectNavOrcaEnv(PointNavOrcaEnv):
    """
    Gym environment for doing the ObjectNav task on the Spot robot in the real world.
    """

    tf_episodic_to_global: np.ndarray = np.eye(4)  # must be set in reset()
    tf_global_to_episodic: np.ndarray = np.eye(4)  # must be set in reset()
    episodic_start_yaw: float = float("inf")  # must be set in reset()
    target_object: str = ""  # must be set in reset()

    def __init__(self, server: RecFromRos2Img, max_gripper_cam_depth: float, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_gripper_cam_depth = max_gripper_cam_depth
        # Get the current date and time
        now = datetime.now()
        #从ros2获取数据
        self.server = server
        self.depth = None
        self.base_to_global = np.eye(4)
        # self.server.run()
        # 在realtimeGI是3.85
        self.scale_xy = 1.0
        # Format it into a string in the format MM-DD-HH-MM-SS
        date_string = now.strftime("%m-%d-%H-%M-%S")
        self._vis_dir = f"{date_string}"
        os.makedirs(f"vis/{self._vis_dir}", exist_ok=True)

    def reset(self, goal: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        assert isinstance(goal, str)
        self.target_object = goal
        # Transformation matrix from where the robot started to the global frame
        # print(self.server.cnt)
        self.base_to_global = self.server.get_transform()
        
        self.depth = self.server.get_float_depth().squeeze()
        for i in range(3):
            self.base_to_global[i,3]= self.base_to_global[i,3]*self.scale_xy
        self.tf_episodic_to_global: np.ndarray = self.base_to_global
        self.tf_episodic_to_global[2, 3] = 0.0  # Make z of the tf 0.0
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        # 弧度制
        start_yaw = np.arctan2(self.base_to_global[1, 0], self.base_to_global[0, 0])
        # self.episodic_start_yaw = np.deg2rad(self.server.array[2])
        self.episodic_start_yaw = start_yaw
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        # Parent class only moves the base; if we want to move the gripper camera,
        # we need to do it here
        vis_imgs = []
        time_id = time.time()
        for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
            img = cv2.cvtColor(action["info"][k], cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"vis/{self._vis_dir}/{time_id}_{k}.png", img)
            if "map" in k:
                img = reorient_rescale_map(img)
            if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
                # Put text in the middle saying "Target not curently detected"
                text = "Target not currently detected"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                cv2.putText(
                    img,
                    text,
                    (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )
            # if k == "annotated_depth":
            #     img = self.depth
            #     img = img * 5
            #     # img拓展到3通道
            #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            vis_imgs.append(img)
        
        resized_vis_imgs = resize_images(vis_imgs, match_dimension="height")
        # 分离左边和右边的图像
        left_img = resized_vis_imgs[0]  # annotated_rgb
        right_imgs = resized_vis_imgs[1:]  # 其他三个图像

        H, W = left_img.shape[:2]
        target_h = H // 3
        target_w = W // 3

        # 调整并填充右侧图像
        right_processed = []
        for img in right_imgs:
            h, w = img.shape[:2]
            scale = min(target_h / h, target_w / w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            resized = cv2.resize(img, (new_w, new_h))
            padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
            y_start = (target_h - new_h) // 2
            x_start = (target_w - new_w) // 2
            padded[y_start:y_start+new_h, x_start:x_start+new_w] = resized
            right_processed.append(padded)

        # 垂直拼接右侧图像并水平合并
        right_combined = np.vstack(right_processed)
        # 修复高度不一致的问题
        if left_img.shape[0] != right_combined.shape[0]:
            # 计算高度差
            height_diff = abs(left_img.shape[0] - right_combined.shape[0])
            # 如果 right_combined 高度较小，则在顶部或底部填充
            if right_combined.shape[0] < left_img.shape[0]:
                padding = ((0, height_diff), (0, 0), (0, 0))  # 仅在高度方向填充
                right_combined = np.pad(right_combined, padding, mode='constant', constant_values=255)
            # 如果 left_img 高度较小，则在顶部或底部填充
            elif left_img.shape[0] < right_combined.shape[0]:
                padding = ((0, height_diff), (0, 0), (0, 0))  # 仅在高度方向填充
                left_img = np.pad(left_img, padding, mode='constant', constant_values=255)
        vis_img = np.hstack([left_img, right_combined])
        # 拼接所有图像并保存
        # vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
        # cv2.imwrite(f"vis/{self._vis_dir}/{time_id}.jpg", vis_img)
        # 显示到窗口（若环境变量允许）
        if os.environ.get("ZSOS_DISPLAY", "0") == "1":
            cv2.imshow("Visualization", cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(1)
        # print(vis_img.shape)
        target_shape = (720, 1280)
        # target_shape = (1080, 1920)
        cv2.imshow("Visualiz", cv2.resize(vis_img, target_shape[::-1]))
        cv2.waitKey(1)

        # 机仅移动底座
        if action["arm_yaw"] == -1:
            # print("仅移动底座")
            return super().step(action)

        # 移动机械臂到预设位置
        # if action["arm_yaw"] == 0:
        #     cmd_id = self.robot.spot.move_gripper_to_point(np.array([0.35, 0.0, 0.3]), np.deg2rad([0.0, 0.0, 0.0]))
        #     self.robot.spot.block_until_arm_arrives(cmd_id, timeout_sec=1.5)
        # # 调整机械臂的关节角度
        # else:
        #     # NOMINAL_ARM_POSE为预设关节角度
        #     new_pose = np.array(NOMINAL_ARM_POSE)
        #     new_pose[0] = action["arm_yaw"]
        #     self.robot.set_arm_joints(new_pose, travel_time=0.5)
        #     time.sleep(0.75)
        done = False
        self._num_steps += 1

        # 奖励固定为0.0，无额外信息
        return self._get_obs(), 0.0, done, {}  # not using info dict yet

    def _get_obs(self) -> Dict[str, Any]:

        nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = self._get_camera_obs()
        robot_xy, robot_heading = self._get_gps(), self._get_compass()
        # robot_xy = robot_xy*self.scale_xy
        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.target_object,
            "obstacle_map_depths": obstacle_map_depths,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }

    def _get_camera_obs(self) -> Tuple[np.ndarray, List, List, List]:
        """
        Poll all necessary cameras on the robot and return their images, focal lengths,
        and transforms to the global frame.
        """
        # **ALL_CAMS**: 包含所有需要轮询的相机源列表（如 "HAND_COLOR", "FRONTLEFT_DEPTH" 等）。
        # ​**POINT_CLOUD_CAMS**: 用于构建点云地图的相机源列表（如前左/前右深度相机）。
        # ​**VALUE_MAP_CAMS**: 用于价值地图的 RGB 相机源列表（如 "FRONTLEFT", "FRONTRIGHT"）。
        srcs: List[str] = ALL_CAMS
        # cam_data = self.robot.get_camera_data(srcs)
        raw_rgb = self.server.get_color()
        raw_depth = self.server.get_float_depth().squeeze()
        self.depth = raw_depth
        # cv2.imshow("raw_depth", raw_depth)
        # cv2.waitKey(1)
        cam_data = {}  
        # 首先更新相机数据和 base_to_global 位姿
        self.base_to_global = self.server.get_transform()
        for i in range(3):
            self.base_to_global[i,3]= self.base_to_global[i,3]*self.scale_xy
        # print(f"_get_camera_obs_tf: {self.base_to_global[0:2, 3]}")

        # (global to episodic) * (base to global) = base to episodic
        # tf = self.tf_global_to_episodic @ self.base_to_global
        tf = np.dot(self.tf_global_to_episodic , self.base_to_global)
        camera_to_base = np.array([[1, 0, 0, 0.325], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # (base to episodic) * (camera to base) = camera to episodic
        # tf = tf @ camera_to_base
        tf = np.dot(tf , camera_to_base)

        # tmp_yaw = np.arctan2(self.base_to_global[1, 0], self.base_to_global[0, 0])
        # print(f"tf_camera_to_global——yaw: {tmp_yaw}")

        for src in ALL_CAMS:
            cam_data[src] = {}
            # a tf that remaps from camera conventions to xyz conventions
            # 这里的rotation_matrix是 相机坐标系 -> xyz坐标系(base坐标系)
            # 相机的z轴（向前） → xyz的x轴（向前）。
            # 相机的x轴（向右） → xyz的y轴（向左，取反）。
            # 相机的y轴（向下） → xyz的z轴（向上，取反）。
            rotation_matrix = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
            # 公共参数
            data = {
                "fx": 240.0 ,
                "fy": 240.0 ,
                "tf_camera_to_global": tf,
            }

            # 深度数据处理
            if src in POINT_CLOUD_CAMS:
                data["image"] = raw_depth.copy()
            # RGB数据处理
            elif src in VALUE_MAP_CAMS:
                # rgb = self._process_rgb(raw_rgb.copy(), src)
                data["image"] = raw_rgb.copy()
            cam_data[src] = data

            img = cam_data[src]["image"]

            # Normalize and filter depth images; don't filter nav depth yet
            if img.dtype == np.uint16:
                if "hand" in src:
                    max_depth = self._max_gripper_cam_depth
                else:
                    max_depth = self._max_body_cam_depth
                img = self._norm_depth(img, max_depth=max_depth)
                # Don't filter nav depth yet
                if "front" not in src:
                    img = filter_depth(img, blur_type=None, recover_nonzero=False)
                cam_data[src]["image"] = img

            if img.dtype == np.uint8:
                if img.ndim == 2 or img.shape[2] == 1:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        min_depth = 0

        # Object map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fx, fy) for each of the cameras in POINT_CLOUD_CAMS
        src = SpotCamIds.HAND_COLOR
        rgb = cam_data[src]["image"]
        hand_depth = np.ones(rgb.shape[:2], dtype=np.float32)
        tf = cam_data[src]["tf_camera_to_global"]
        max_depth = self._max_gripper_cam_depth
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
        object_map_rgbd = [(rgb, hand_depth, tf, min_depth, max_depth, fx, fy)]

        # Nav depth requires the front two camera images, and they must be rotated
        # to be upright
        f_left = SpotCamIds.FRONTLEFT_DEPTH
        f_right = SpotCamIds.FRONTRIGHT_DEPTH
        # 重新定向图像（旋转/平移）以对齐坐标系。
        # nav_cam_data = self.robot.reorient_images({k: cam_data[k]["image"] for k in [f_left, f_right]})
        # nav_depth = np.hstack([nav_cam_data[f_right], nav_cam_data[f_left]])
        
        nav_depth = cam_data[f_left]["image"]
        # nav_depth = np.squeeze(nav_depth)
        # print(nav_depth.shape,"---", cam_data[f_right]["image"].shape)

        # nav_depth = nav_depth[100:-100,:]  
        nav_depth = filter_depth(nav_depth, blur_type=None, set_black_value=1.0)

        # Obstacle map output is a list of (depth, tf, min_depth, max_depth, fx, fy,
        # topdown_fov) for each of the cameras in POINT_CLOUD_CAMS
        obstacle_map_depths = []
        if self._num_steps <= 10:
            srcs = POINT_CLOUD_CAMS.copy()
        else:
            srcs = POINT_CLOUD_CAMS[:2]
        for src in srcs:
            depth = cam_data[src]["image"]
            depth = depth[100:-60,:]  
            fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
            tf = cam_data[src]["tf_camera_to_global"]
            # print(f"obstacle_map_depths_pos:{tf[0:2, 3]}")
            if src in [SpotCamIds.FRONTLEFT_DEPTH, SpotCamIds.FRONTRIGHT_DEPTH]:
                fov = get_fov(fy, depth.shape[0])
            else:
                fov = get_fov(fx, depth.shape[1])
            src_data = (depth, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
            obstacle_map_depths.append(src_data)

        tf = cam_data[SpotCamIds.HAND_COLOR]["tf_camera_to_global"]
        fx, fy = (
            cam_data[SpotCamIds.HAND_COLOR]["fx"],
            cam_data[SpotCamIds.HAND_COLOR]["fy"],
        )
        fov = get_fov(fx, cam_data[src]["image"].shape[1])
        src_data = (None, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
        obstacle_map_depths.append(src_data)

        # Value map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fov) for each of the cameras in VALUE_MAP_CAMS
        value_map_rgbd = []
        value_cam_srcs: List[str] = VALUE_MAP_CAMS + ["hand_depth_estimated"]
        # RGB cameras are at even indices, depth cameras are at odd indices
        value_rgb_srcs = value_cam_srcs[::2]
        value_depth_srcs = value_cam_srcs[1::2]
        for rgb_src, depth_src in zip(value_rgb_srcs, value_depth_srcs):
            rgb = cam_data[rgb_src]["image"]
            if depth_src == "hand_depth_estimated":
                depth = hand_depth
            else:
                depth = cam_data[src]["image"]
            fx = cam_data[rgb_src]["fx"]
            tf = cam_data[rgb_src]["tf_camera_to_global"]
            fov = get_fov(fx, rgb.shape[1])
            src_data = (rgb, depth, tf, min_depth, max_depth, fov)  # type: ignore
            value_map_rgbd.append(src_data)

        return nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd

    def _get_gps(self) -> np.ndarray:
        """
        Get the (x, y) position of the robot's base in the episode frame. x is forward,
        y is left.
        """
        # global_xy = self.robot.xy_yaw[0]
        # global_xy = self.server.array[3:5]
        global_xy = self.base_to_global[:2, 3]
        start_xy = self.tf_episodic_to_global[:2, 3]
        offset = global_xy - start_xy
        rotation_matrix = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        episodic_xy = rotation_matrix @ offset
        # print(f"start_xy:{start_xy} | global_xy:{global_xy} | offset:{offset}")
        # print(f"_get_gps:{global_xy},{episodic_xy}")
        return episodic_xy

    def _get_compass(self) -> float:
        """
        Get the yaw of the robot's base in the episode frame. Yaw is measured in radians
        counterclockwise from the z-axis.
        """
        # global_yaw = self.robot.xy_yaw[1]
        # global_yaw = np.deg2rad(self.server.array[2])
        # #self.base_to_global是一个4x4矩阵，通过此矩阵获得yaw，这里的yaw度数是弧度制
        global_yaw = np.arctan2(self.base_to_global[1, 0], self.base_to_global[0, 0])
        episodic_yaw = wrap_heading(global_yaw - self.episodic_start_yaw)
        # print("episodic_yaw", episodic_yaw)
        # print("_get_compass_yaw:", global_yaw)
        return episodic_yaw
