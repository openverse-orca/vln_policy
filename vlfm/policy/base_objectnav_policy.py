# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import requests, time
from hydra.core.config_store import ConfigStore
from torch import Tensor
import matplotlib.pyplot as plt

from vlfm.mapping.object_point_cloud_map import ObjectPointCloudMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import get_fov, rho_theta
from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.coco_classes import COCO_CLASSES
from vlfm.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from vlfm.vlm.sam import MobileSAMClient
from  vlfm.vlm.yolov7 import YOLOv7Client

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from  vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass



def display_grayscale(image, waittime: int=100):
    image = np.expand_dims(image, axis=-1)
    img_bgr = np.repeat(image, 3, 2)
    # print(f"Image data type: {img_bgr.dtype}")
    # print(img_bgr.size, img_bgr.shape)
    cv2.imshow("Depth Sensor", img_bgr)
    return cv2.waitKey(waittime)

class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _object_masks: Union[np.ndarray, Any] = None  # set by ._update_object_map()
    _stop_action: Union[Tensor, Any] = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}
    _non_coco_caption = ""
    _load_yolo: bool = True

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        use_vqa: bool = False,
        vqa_prompt: str = "Is this ",
        coco_threshold: float = 0.8,
        non_coco_threshold: float = 0.4,
        url: str = "http://localhost:"+str(15533)+"/rec",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        url = "http://192.168.1.184:"+str(15533)+"/rec"
        self.url = url
        # 这里导入了这三个模型
        self._object_detector = GroundingDINOClient(port=int(os.environ.get("GROUNDING_DINO_PORT", "12181")))
        self._coco_object_detector = YOLOv7Client(port=int(os.environ.get("YOLOV7_PORT", "12184")))
        self._mobile_sam = MobileSAMClient(port=int(os.environ.get("SAM_PORT", "12183")))
        self._use_vqa = use_vqa
        if use_vqa:
            self._vqa = BLIP2Client(port=int(os.environ.get("BLIP2_PORT", "12185")))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(erosion_size=object_map_erosion_size)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius


        visualize = True
        self._visualize = visualize
        self._vqa_prompt = vqa_prompt
        self._coco_threshold = coco_threshold
        self._non_coco_threshold = non_coco_threshold

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )
        # print("BaseObjectNavPolicy init finish--------")
        
        self.i_error_rho = 0
        self.i_error_theta = 0
        self.previous_error_rho = 0
        self.previous_error_theta = 0

        self.previous_goal = None
        

    def _reset(self) -> None:
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True

    def _simple_pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        # print("rho",rho)
        # print("theta",theta)
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            print("STOP!!!", self._stop_action)
            return self._stop_action

        # if np.abs(theta) < (0.3 if rho > 1.5 else 0.6):
        if np.abs(theta) < (0.03 if rho > 1.5 else 0.06):
            return "straight", torch.tensor([[1]])
        if theta > 0:
            return "left", torch.tensor([[theta]])
        else:
            return "right", torch.tensor([[theta]])
        
    def _pid_pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)


        # # log rho, theta in csv file
        # with open("rho_theta.csv", "a") as f:
        #     print("writing")
        #     f.write(f"{rho},{theta}\n")
        # print(rho, theta)
        
        target_rho = 1
        target_theta = 0
        
        kp_rho = 1
        ki_rho = 0
        kd_rho = 0
        
        kp_theta = 1.5
        ki_theta = 0.01
        kd_theta = 0
        
        p_error_rho = rho - target_rho
        p_error_theta = theta - target_theta
        
        self.i_error_rho += p_error_rho
        self.i_error_theta += p_error_theta
        
        d_error_rho = p_error_rho - self.previous_error_rho
        d_error_theta = p_error_theta - self.previous_error_theta
        
        self.previous_error_rho = d_error_rho
        self.previous_error_theta = d_error_theta
        
        signal_rho = kp_rho * p_error_rho + ki_rho * self.i_error_rho + kd_rho * d_error_rho
        signal_theta = kp_theta * p_error_theta + ki_theta * self.i_error_theta + kd_theta * d_error_theta
        
        # print("p_error_rho: ", p_error_rho, "p_error_theta: ", p_error_theta)
        # print("d_error_rho: ", d_error_rho, "d_error_theta: ", d_error_theta)
        # print("i_error_rho: ", self.i_error_rho, "i_error_theta: ", self.i_error_theta)
        # print("signal_rho: ", signal_rho, "signal_theta: ", signal_theta)
        
        return "pid", torch.tensor([[signal_rho, signal_theta]])
        
        
             

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Any:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal_perception = self._get_target_object_location(robot_xy)
        goal_cheating = observations["person_pos_xy"]

        
        # store in csv
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if self._object_map.has_object(self._target_object):
            cloud = self._object_map.get_target_cloud(self._target_object)

            ax.scatter(cloud[:, 0], cloud[:, 1], s=10)
        if goal_perception is not None:
            ax.scatter([goal_perception[0]], [goal_perception[1]], c="blue")
        ax.scatter([goal_cheating[0]], [goal_cheating[1]], c="black")
        ax.text(goal_cheating[0], goal_cheating[1], 'Cheating', fontsize=8, color='black')
        ax.scatter([self._observations_cache["robot_xy"][0]], [self._observations_cache["robot_xy"][1]], c="red")
        ax.text(robot_xy[0], robot_xy[1], 'Robot', fontsize=8, color='red')
        # Ensure fixed axis ranges
        # ax.set_xlim(-5, 0)   # Change these limits as you need
        # ax.set_ylim(-15, 0)
        plt.savefig(f"pics/global_cloud{int(time.time())}.png")

        # with open("goals.csv", "a") as f:
        #     if goal_perception is None:
        #         goal_perception = [0, 0]
        #     f.write(f"{goal_perception[0]}, {goal_perception[1]}, {goal_cheating[0]},{goal_cheating[1]}\n")
            
        
        if goal_perception is None:
            if self.previous_goal is None:
                goal = None
            else:
                goal = self.previous_goal
                # print(robot_xy)
                # print(self.previous_goal)
        else:
            goal = goal_perception

        self.previous_goal = goal

        self._object_map.reset()


        # if not self._done_initializing:  # Initialize
        #     mode = "initialize"
        #     pointnav_action = self._initialize()
        # elif goal is None:  # Haven't found target object yet
        #     mode = "explore"
        #     pointnav_action = self._explore(observations)
        # else:
        #     mode = "navigate"
        #     pointnav_action = self._pointnav(goal[:2], stop=True)
        
        if goal is not None:
            # mode, pointnav_action = self._simple_pointnav(goal[:2], stop=True)
            mode, pointnav_action = self._pid_pointnav(goal[:2], stop=True)
        else:
            mode = "initialize"
            pointnav_action = self._initialize()  

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        # print(f"origin action_numpy:{action_numpy}")
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")

        data= {
            "step": self._num_steps,
            "mode": mode,
            "action": action_numpy.tolist(),
        }
        try:
            response = requests.post(self.url, json=data)
        except Exception as e:
            # print("___^^___",e)
            pass
        self._policy_info.update(self._get_policy_info(detections[0]))
        self._num_steps += 1



        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position: np.ndarray) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object.split("|")[0],
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._object_map.has_object(self._target_object),
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            print("9012-0349-1082341-9248-=-==-=-=--==-=-=-----=-")
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            annotated_rgb = cv2.drawContours(detections.annotated_frame, contours, -1, (255, 0, 0), 2)
            annotated_depth = cv2.drawContours(annotated_depth, contours, -1, (255, 0, 0), 2)
        else:
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB)

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        target_classes = self._target_object.split("|")
        # has_coco = any(c in COCO_CLASSES for c in target_classes) and self._load_yolo
        has_coco = any(c in COCO_CLASSES for c in target_classes) 

        has_non_coco = any(c not in COCO_CLASSES for c in target_classes)

        detections = (
            self._coco_object_detector.predict(img)
            if has_coco
            else self._object_detector.predict(img, caption=self._non_coco_caption)
        )
        detections.filter_by_class(target_classes)
        det_conf_threshold = self._coco_threshold if has_coco else self._non_coco_threshold
        detections.filter_by_conf(det_conf_threshold)

        if has_coco and has_non_coco and detections.num_detections == 0:
            # Retry with non-coco object detector
            detections = self._object_detector.predict(img, caption=self._non_coco_caption)
            detections.filter_by_class(target_classes)
            detections.filter_by_conf(self._non_coco_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        # print(f"Base_object_nav_policy---robot_xy: {robot_xy}")
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor([[rho, theta]], device="cuda", dtype=torch.float32)
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            print("stop", self._stop_action)
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)

        # display_grayscale(depth)
        max_depth=1
        min_depth = 0
    
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)

        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array([width, height, width, height])
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())

            # If we are using vqa, then use the BLIP2 model to visually confirm whether
            # the contours are actually correct.

            if self._use_vqa:
                contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                annotated_rgb = cv2.drawContours(rgb.copy(), contours, -1, (255, 0, 0), 2)
                question = f"Question: {self._vqa_prompt}"
                if not detections.phrases[idx].endswith("ing"):
                    question += "a "
                question += detections.phrases[idx] + "? Answer:"
                answer = self._vqa.ask(annotated_rgb, question)
                if not answer.lower().startswith("yes"):
                    continue

            self._object_masks[object_mask > 0] = 1

            self._object_map.update_map(
                self._target_object,
                depth,
                object_mask,
                tf_camera_to_episodic,
                min_depth,
                max_depth,
                fx,
                fy,
            )

        cone_fov = get_fov(fx, depth.shape[1])
        self._object_map.update_explored(tf_camera_to_episodic, max_depth, cone_fov)

        # display_img
        return detections

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class VLFMConfig:
    name: str = "HabitatITMPolicy"
    text_prompt: str = "Seems like there is a target_object ahead."
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    use_vqa: bool = False
    vqa_prompt: str = "Is this "
    coco_threshold: float = 0.8
    non_coco_threshold: float = 0.4
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(VLFMConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="vlfm_config_base", node=VLFMConfig())
