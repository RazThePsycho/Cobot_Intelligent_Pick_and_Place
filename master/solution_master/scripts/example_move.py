#!/usr/bin/env python3


import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from cv_bridge import CvBridge
from rospkg import RosPack
import os
import tf2_ros
import cv2
import open3d as o3d
import ultralytics
from ultralytics import YOLO
from utils.scene_detector import SceneDetector
from utils.grip_pose_estimator import GripPoseEstimator
from utils.utils import (
    create_inspection_so3_in_global_frame,
    calculate_so3_oriented_to_target,
    generate_ellipse_trajectory,
    check_distance_and_angle,
)
import traceback
import sys

ultralytics.checks()

rp = RosPack()

GRIPPER_JOINT_NAMES = ["gripper_finger1_joint"]
MANIPULATOR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

HOME_JOINT_POSITIONS = [
    0,
    -1.7782234350751653,
    1.6973525342276288,
    -1.4109909996799024,
    -1.6052999031392083,
    -0.31743525548842033,
]


PER_SIDE_POINTS = np.array(
    [
        [
            0.3,
            -0.0,
            1.1,
            -0.6447705311538084,
            0.7165366385001714,
            -0.183031483277461,
            0.19325031428632358,
        ],
        [
            0.2648204544910632,
            0.00030229216542194737,
            0.9980174416225138,
            -0.7001528699810571,
            0.7124624113456665,
            -0.03097791104407552,
            0.03498056752187769,
        ],
    ]
)

GRIP_TRANSFORM = [0, 0, 0.15, 0, 0, 0, 1]


class YOLObjectDetector:
    def __init__(
        self,
        chkpt_path=os.path.join(
            rp.get_path("solution_master"),
            "checkpoints/yolo_m_1200_epochs/weights/best.pt",
        ),
    ) -> None:
        self.model = YOLO(chkpt_path)
        self.bridge = CvBridge()
        self.result_image_publisher = rospy.Publisher(
            "/yolo/result_image", Image, queue_size=10
        )

    def predict(self, image: np.ndarray, return_image: bool = False):
        result = self.model(image)[0]
        if return_image:
            return result, result.plot()
        else:
            return result

    def predict_and_publish_result(self):
        image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image), "bgr8"
        )
        result, result_image = self.predict(image, return_image=True)
        self.result_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(result_image, "bgr8")
        )


class ArmController:
    def __init__(self) -> None:
        moveit_commander.roscpp_initialize(sys.argv)
        self.manipulator_commander = moveit_commander.MoveGroupCommander("robot_arm")
        self.gripper_commander = moveit_commander.MoveGroupCommander("robot_gripper")
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.yolo_model = YOLObjectDetector()
        self.grip_pose_estimator = GripPoseEstimator()
        self.scene_detector = SceneDetector()
        self.poses_joint_states_correspondence = np.load(
            os.path.join(rp.get_path("solution_master"), "correspondence/poses.npy"),
            allow_pickle=True,
        ).item()
        self.trashcan_place_pose = None
        self.tetrapck_place_pose = None
        self.start_position = None
        self.start_joint_states = None
        self.trashcan_place_joint_states = None
        self.tetrapck_place_joint_states = None
        self.table_se3 = None

    def get_manipulator_joint_states(self) -> np.ndarray:
        return np.array(self.manipulator_commander.get_current_joint_values())

    def init_environment_poses(self, num_rotations=6):
        """Rotating around for obtaining point cloud the scene

        Args:
            num_rotations (int, optional): amount of point around z axis which one would be moved through. Defaults to 10.
        """
        rz_rotations = R.from_euler(
            "z",
            [i * 2 * np.pi / num_rotations - np.pi / 2 for i in range(num_rotations)],
            degrees=False,
        )
        all_points = []
        for i in range(2):
            point = PER_SIDE_POINTS[i]
            all_points.append(
                np.concatenate(
                    [
                        rz_rotations.apply(point[:3]),
                        (rz_rotations * R.from_quat(point[3:])).as_quat(),
                    ],
                    axis=1,
                )[:: -1 if i % 2 else 1]
            )
        points_to_visit = np.concatenate(all_points, axis=0)
        pcd_list = []
        for i, point in enumerate(points_to_visit):
            self.move_trajectory([point])
            pcd_list.append(self.get_current_pointcloud())
        detected_objects = self.scene_detector.compute_scene_objects_poses(pcd_list)
        self.trashcan_place_pose = np.array(
            [
                *detected_objects["blue_trashbox_se3"][:3, 3],
                *R.from_matrix(detected_objects["blue_trashbox_se3"][:3, :3]).as_quat(),
            ]
        )
        self.tetrapck_place_pose = np.array(
            [
                *detected_objects["orange_trashbox_se3"][:3, 3],
                *R.from_matrix(
                    detected_objects["orange_trashbox_se3"][:3, :3]
                ).as_quat(),
            ]
        )
        self.plane_coefficients = detected_objects["plane_coefficients"]

        self.table_se3 = detected_objects["table_se3"]
        self.grip_pose_estimator.table_pose = self.table_se3

    def get_current_pointcloud(self):
        rgb_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image), "bgr8"
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/depth/image_raw", Image),
            desired_encoding="passthrough",
        )
        depth_image = np.array(np.rint(depth_image * 1000), dtype=np.uint16)
        depth_threshold = 150
        depth_image[depth_image < depth_threshold] = 0
        camera_intrinsic = np.array(
            rospy.wait_for_message("/hand_eye/camera/depth/camera_info", CameraInfo).K
        ).reshape((3, 3))
        exctrinsic_pose = self.tf_buffer.lookup_transform(
            "world", "hand_eye_depth_optical_frame", rospy.Time()
        )
        exctrinsic_pose = [
            exctrinsic_pose.transform.translation.x,
            exctrinsic_pose.transform.translation.y,
            exctrinsic_pose.transform.translation.z,
            exctrinsic_pose.transform.rotation.x,
            exctrinsic_pose.transform.rotation.y,
            exctrinsic_pose.transform.rotation.z,
            exctrinsic_pose.transform.rotation.w,
        ]
        metadata = {
            "camera_intrinsic": camera_intrinsic,
            "exctrinsic_pose": exctrinsic_pose,
        }
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=camera_intrinsic[0, 0],
            fy=camera_intrinsic[1, 1],
            cx=camera_intrinsic[0, 2],
            cy=camera_intrinsic[1, 2],
        )
        o3d_rgb_image = o3d.geometry.Image(rgb_image)
        o3d_depth_image = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb_image,
            o3d_depth_image,
            convert_rgb_to_intensity=False,
            depth_scale=1000,
        )
        camera_transform = np.eye(4)
        camera_transform[:3, 3] = metadata["exctrinsic_pose"][:3]
        camera_transform[:3, :3] = R.from_quat(
            metadata["exctrinsic_pose"][3:]
        ).as_matrix()
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d_camera_intrinsic
        )
        pcd = pcd.transform(camera_transform)
        return pcd

    def save_point_cloud(self, filename: str):
        package_path = os.path.join(
            rp.get_path("solution_master"), "debug_folder/debug_old"
        )
        rgb_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image), "bgr8"
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/depth/image_raw", Image),
            desired_encoding="passthrough",
        )
        # converting to millimeters
        depth_image = np.array(np.rint(depth_image * 1000), dtype=np.uint16)

        camera_intrinsic = np.array(
            rospy.wait_for_message("/hand_eye/camera/depth/camera_info", CameraInfo).K
        ).reshape((3, 3))
        exctrinsic_pose = self.tf_buffer.lookup_transform(
            "world", "hand_eye_depth_optical_frame", rospy.Time()
        )
        # print(exctrinsic_pose)
        exctrinsic_pose = [
            exctrinsic_pose.transform.translation.x,
            exctrinsic_pose.transform.translation.y,
            exctrinsic_pose.transform.translation.z,
            exctrinsic_pose.transform.rotation.x,
            exctrinsic_pose.transform.rotation.y,
            exctrinsic_pose.transform.rotation.z,
            exctrinsic_pose.transform.rotation.w,
        ]
        # print(f"Extrinsic: {exctrinsic_pose}")
        # Saving all the data
        cv2.imwrite(os.path.join(package_path, f"rgb_{filename}.png"), rgb_image)
        cv2.imwrite(os.path.join(package_path, f"depth_{filename}.png"), depth_image)
        metadata = {
            "camera_intrinsic": camera_intrinsic,
            "exctrinsic_pose": exctrinsic_pose,
        }
        np.save(os.path.join(package_path, f"metadata_{filename}.npy"), metadata)

        pass

    def save_rgb_and_depth(self):
        save_path = os.path.join(rp.get_path("solution_master"), "debug_folder")
        rgb_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image), "bgr8"
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/depth/image_raw", Image),
            desired_encoding="passthrough",
        )
        depth_image = np.array(np.rint(depth_image * 1000), dtype=np.uint16)
        camera_intrinsic = np.array(
            rospy.wait_for_message("/hand_eye/camera/depth/camera_info", CameraInfo).K
        ).reshape((3, 3))
        exctrinsic_pose = self.tf_buffer.lookup_transform(
            "world", "hand_eye_depth_optical_frame", rospy.Time()
        )
        exctrinsic_pose = [
            exctrinsic_pose.transform.translation.x,
            exctrinsic_pose.transform.translation.y,
            exctrinsic_pose.transform.translation.z,
            exctrinsic_pose.transform.rotation.x,
            exctrinsic_pose.transform.rotation.y,
            exctrinsic_pose.transform.rotation.z,
            exctrinsic_pose.transform.rotation.w,
        ]
        cv2.imwrite(os.path.join(save_path, "rgb.png"), rgb_image)
        cv2.imwrite(os.path.join(save_path, "depth.png"), depth_image)
        metadata = {
            "camera_intrinsic": camera_intrinsic,
            "exctrinsic_pose": exctrinsic_pose,
        }
        np.save(os.path.join(save_path, "metadata.npy"), metadata)

    def prepare_data_for_grasping(self, target_class):
        rgb_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image), "bgr8"
        )
        depth_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/depth/image_raw", Image),
            desired_encoding="passthrough",
        )
        depth_image = np.array(np.rint(depth_image * 1000), dtype=np.uint16)
        depth_threshold = 150
        depth_image[depth_image < depth_threshold] = 0
        camera_intrinsic = np.array(
            rospy.wait_for_message("/hand_eye/camera/depth/camera_info", CameraInfo).K
        ).reshape((3, 3))
        exctrinsic_pose = self.tf_buffer.lookup_transform(
            "world", "hand_eye_depth_optical_frame", rospy.Time()
        )
        exctrinsic_pose = [
            exctrinsic_pose.transform.translation.x,
            exctrinsic_pose.transform.translation.y,
            exctrinsic_pose.transform.translation.z,
            exctrinsic_pose.transform.rotation.x,
            exctrinsic_pose.transform.rotation.y,
            exctrinsic_pose.transform.rotation.z,
            exctrinsic_pose.transform.rotation.w,
        ]

        metadata = {
            "camera_intrinsic": camera_intrinsic,
            "exctrinsic_pose": exctrinsic_pose,
        }

        yolo_predictions, image = self.yolo_model.predict(rgb_image, return_image=True)
        self.yolo_model.result_image_publisher.publish(
            self.bridge.cv2_to_imgmsg(image, "bgr8")
        )
        yolo_predictions = [
            prediction
            for prediction in yolo_predictions
            if prediction.boxes.cls.item() == target_class
        ]
        needed_yolo_result = max(
            yolo_predictions, key=lambda x: x.masks.data.sum().item()
        )
        return [needed_yolo_result], rgb_image, depth_image, metadata

    @staticmethod
    def grip_point_from_tool(grip_point: np.ndarray) -> np.ndarray:
        grip_rotation = R.from_quat(grip_point[3:])
        return [
            *(grip_point[:3] + grip_rotation.apply(GRIP_TRANSFORM[:3])),
            *grip_point[3:],
        ]

    @staticmethod
    def tool_from_grip_point(tool_point: np.ndarray) -> np.ndarray:
        tool_rotation = R.from_quat(tool_point[3:])
        return [
            *(tool_point[:3] - tool_rotation.apply(GRIP_TRANSFORM[:3])),
            *tool_point[3:],
        ]

    def get_corresponded_joint_states(self, pose: np.ndarray):
        idx = np.linalg.norm(
            np.array(self.poses_joint_states_correspondence["poses"])[:, :3] - pose[:3],
            axis=1,
        ).argmin()
        return self.poses_joint_states_correspondence["joint_states"][idx]

    @staticmethod
    def convert_se3_to_array(se3):
        return np.array([*se3[:3, 3], *R.from_matrix(se3[:3, :3]).as_quat()])

    def strategy(self):

        self.start_position = self.convert_se3_to_array(
            create_inspection_so3_in_global_frame(
                [0, 0, 0.5], [0.2, 0, -0.5], self.table_se3
            )
        )
        self.start_joint_states = self.get_corresponded_joint_states(
            self.start_position
        )
        self.trashcan_place_joint_states = self.get_corresponded_joint_states(
            self.trashcan_place_pose
        )
        self.tetrapck_place_joint_states = self.get_corresponded_joint_states(
            self.tetrapck_place_pose
        )

        self.move_joints(self.start_joint_states)

        points_to_inspect = [
            self.convert_se3_to_array(
                create_inspection_so3_in_global_frame(
                    [0.1, t, 0.3], [0, 0, -0.5], self.table_se3
                )
            )
            for t in np.linspace(-0.3, 0.3, 10)
        ] + [
            self.convert_se3_to_array(
                create_inspection_so3_in_global_frame(
                    [-0.05, t, 0.4], [0, 0, -0.5], self.table_se3
                )
            )
            for t in np.linspace(-0.3, 0.3, 10)
        ]

        for point in points_to_inspect:
            self.move_trajectory([point])
            try:
                self.detect_and_grip_object()
            except Exception as e:
                print(traceback.format_exc())
                # or
                print(sys.exc_info()[2])

    def move_to_pose_in_joint_space(self, pose: np.ndarray):

        np_point = self.tool_from_grip_point(pose)
        pose = geometry_msgs.msg.Pose()
        pose.position.x = np_point[0]
        pose.position.y = np_point[1]
        pose.position.z = np_point[2]
        pose.orientation.x = np_point[3]
        pose.orientation.y = np_point[4]
        pose.orientation.z = np_point[5]
        pose.orientation.w = np_point[6]

        # self.manipulator_commander.set_pose_target(pose)
        self.manipulator_commander.set_joint_value_target(pose, False)

        success = self.manipulator_commander.go(wait=True)

        # Ensures that there is no residual movement
        self.manipulator_commander.stop()

        # Clear your targets after planning with poses.
        self.manipulator_commander.clear_pose_targets()
        return success

    def is_object_grasped(self):
        depth_image = self.bridge.imgmsg_to_cv2(
            rospy.wait_for_message("/hand_eye/camera/depth/image_raw", Image),
            desired_encoding="passthrough",
        )
        depth_image = np.array(np.rint(depth_image * 1000), dtype=np.uint16)
        mean = np.mean(depth_image[350:480, 320 - 50 : 320 + 50])
        if mean < 150:
            return True
        return False

    def detect_and_grip_object(self):
        target_class = 1 if np.random.rand() > 0.5 else 0
        self.change_gripper_opening(0.0)
        trajectory_to_move, classes = self.grip_pose_estimator.calculate_grip_poses(
            *self.prepare_data_for_grasping(target_class)
        )
        trajectory_of_vectors = [
            [*pose[:3, 3], *R.from_matrix(pose[:3, :3]).as_quat()]
            for pose in trajectory_to_move
        ]
        self.move_trajectory([trajectory_of_vectors[0]])
        trajectory_to_move, classes = self.grip_pose_estimator.calculate_grip_poses(
            *self.prepare_data_for_grasping(target_class)
        )
        trajectory_of_vectors = [
            [*pose[:3, 3], *R.from_matrix(pose[:3, :3]).as_quat()]
            for pose in trajectory_to_move
        ]
        self.manipulator_commander.set_max_velocity_scaling_factor(0.1)
        self.manipulator_commander.set_max_acceleration_scaling_factor(0.1)
        self.move_trajectory(trajectory_of_vectors)
        print("OBJECT CLASS: ", classes)
        obj_class = classes[0].item()
        if obj_class == 1:
            # trashcan
            self.change_gripper_opening(0.3)
        elif obj_class == 0:
            # tetrapack
            self.change_gripper_opening(0.2)
        self.manipulator_commander.set_max_velocity_scaling_factor(1.0)
        self.manipulator_commander.set_max_acceleration_scaling_factor(1.0)
        self.move_trajectory([trajectory_of_vectors[0]])
        if self.is_object_grasped():
            print("!!!!!!Object is grasped!!!!!")
            # self.move_trajectory(
            #     [self.get_current_grip_pose() + np.array([0, 0, 0.3, 0, 0, 0, 0])]
            # )
            self.move_joints(self.start_joint_states)
            if obj_class == 1:
                self.move_joints(self.trashcan_place_joint_states)
                self.move_trajectory([self.trashcan_place_pose])
                self.change_gripper_opening(0.0)
                self.move_joints(self.trashcan_place_joint_states)
            else:
                self.move_joints(self.tetrapck_place_joint_states)
                self.move_trajectory([self.tetrapck_place_pose])
                self.change_gripper_opening(0.0)
                self.move_joints(self.tetrapck_place_joint_states)
            self.move_joints(self.start_joint_states)
            # self.move_to_pose_in_joint_space(self.start_position)

    def move_to_home(self):
        joints = sensor_msgs.msg.JointState()
        joints.name = MANIPULATOR_JOINT_NAMES
        joints.position = HOME_JOINT_POSITIONS
        success = self.manipulator_commander.go(joints, wait=True)
        self.manipulator_commander.stop()
        self.manipulator_commander.clear_pose_targets()
        return success

    def move_joints(self, target_joints: np.ndarray):
        self.manipulator_commander.set_max_velocity_scaling_factor(0.3)
        joints = sensor_msgs.msg.JointState()
        joints.name = MANIPULATOR_JOINT_NAMES
        joints.position = target_joints.tolist()
        success = self.manipulator_commander.go(joints, wait=True)
        self.manipulator_commander.stop()
        self.manipulator_commander.clear_pose_targets()
        self.manipulator_commander.set_max_velocity_scaling_factor(1.0)
        # rospy.sleep(1)
        return success

    def get_current_grip_pose(self) -> np.ndarray:
        current_pose = np.array(
            pose_to_list(self.manipulator_commander.get_current_pose().pose)
        )
        return self.grip_point_from_tool(current_pose)

    def collect_scene_point_cloud(self, num_rotations=12):
        """Rotating around for obtaining point cloud the scene

        Args:
            num_rotations (int, optional): amount of point around z axis which one would be moved through. Defaults to 10.
        """
        rz_rotations = R.from_euler(
            "z",
            [i * 2 * np.pi / num_rotations - np.pi / 2 for i in range(num_rotations)],
            degrees=False,
        )
        all_points = []
        for i in range(2):
            point = PER_SIDE_POINTS[i]
            all_points.append(
                np.concatenate(
                    [
                        rz_rotations.apply(point[:3]),
                        (rz_rotations * R.from_quat(point[3:])).as_quat(),
                    ],
                    axis=1,
                )[:: -1 if i % 2 else 1]
            )
        points_to_visit = np.concatenate(all_points, axis=0)
        for i, point in enumerate(points_to_visit):
            self.move_trajectory([point])
            self.save_point_cloud(i)

    def collect_dataset_samples(self):
        save_folder = os.path.join(rp.get_path("solution_master"), "rgb_samples")
        idx = len(os.listdir(save_folder))

        se3_to_visit = generate_ellipse_trajectory(
            np.array([0.4, 0.0, 0.45]),
            num_points=20,
            x_center=0.35,
            y_center=0.0,
            radius_x=0.1,
            radius_y=0.3,
            z_coordinate=0.95,
        ) + generate_ellipse_trajectory(
            np.array([0.3, 0.0, 0.6]),
            num_points=20,
            x_center=0.35,
            y_center=0.0,
            radius_x=0.1,
            radius_y=0.3,
            z_coordinate=0.75,
        )
        for se3 in se3_to_visit:
            move_target = np.array([*se3[:3, 3], *R.from_matrix(se3[:3, :3]).as_quat()])
            success = self.move_trajectory([move_target])
            # saving image
            if success:
                self.yolo_model.predict_and_publish_result()
            #     cv2.imwrite(
            #         os.path.join(save_folder, f"rgb_{idx}.png"),
            #         self.bridge.imgmsg_to_cv2(
            #             rospy.wait_for_message("/hand_eye/camera/rgb/image_raw", Image),
            #             "bgr8",
            #         ),
            #     )
            #     idx += 1

    def check_points(self, trajectory_list):
        target = trajectory_list[0]
        current_pose = np.array(self.get_current_grip_pose())
        check, angle = check_distance_and_angle(
            current_pose[0], current_pose[1], target[0], target[1], 0.3
        )
        if check:
            necessary_rotation = R.from_euler("z", [angle / 2], degrees=True)
            addition_pose = np.array(
                [*necessary_rotation.apply(current_pose[:3])[0], *current_pose[3:]]
            )
            trajectory_list.insert(0, addition_pose)
            addition_pose_2 = np.array(
                [*necessary_rotation.inv().apply(target[:3])[0], *target[3:]]
            )
            trajectory_list.insert(1, addition_pose_2)

        return trajectory_list

    def move_trajectory(self, trajectory: list, wait: bool = True) -> bool:
        """moving through trajectory with a grip point target

        Args:
            trajectory (list): list of point (grip coordinates)
            wait (bool, optional): is waiting for movement. Defaults to True.

        Returns:
            bool: success of movement
        """
        # trajectory = self.check_points(trajectory)

        waypoints = []
        for point in trajectory:
            np_point = self.tool_from_grip_point(point)
            pose = geometry_msgs.msg.Pose()
            pose.position.x = np_point[0]
            pose.position.y = np_point[1]
            pose.position.z = np_point[2]
            pose.orientation.x = np_point[3]
            pose.orientation.y = np_point[4]
            pose.orientation.z = np_point[5]
            pose.orientation.w = np_point[6]
            # waypoints.append(copy.deepcopy(pose))
            waypoints.append(pose)
        (plan, fraction) = self.manipulator_commander.compute_cartesian_path(
            waypoints, 0.01, 0.0
        )
        print(f"Fraction: {fraction}")
        if fraction < 0.9:
            return False
        success = self.manipulator_commander.execute(plan, wait=wait)
        self.manipulator_commander.stop()
        self.manipulator_commander.clear_pose_targets()
        return success

    def move_to_point(self, point: np.ndarray, wait: bool = True) -> bool:

        assert point.shape == (6,)
        # print(current_pose):
        current_pose = np.array(
            pose_to_list(self.manipulator_commander.get_current_pose().pose)
        )

    def move_to_inspect_point(self):
        so3_target = calculate_so3_oriented_to_target(
            np.array([0.4, 0.0, 0.8]), np.array([0.2, 0.0, 0.45])
        )
        target_vector = np.array(
            [*so3_target[:3, 3], *R.from_matrix(so3_target[:3, :3]).as_quat()]
        )
        self.move_trajectory([target_vector])
        #  se3_to_visit = generate_ellipse_trajectory(
        #     np.array([0.4, 0.0, 0.45]),
        #     num_points=20,
        #     x_center=0.35,
        #     y_center=0.0,
        #     radius_x=0.1,
        #     radius_y=0.3,
        #     z_coordinate=0.95,

    def change_gripper_opening(
        self, close_position: float = 0.0, wait: bool = True
    ) -> bool:
        assert isinstance(close_position, float) and 0.0 <= close_position <= 0.8
        target_joints = sensor_msgs.msg.JointState()
        target_joints.name = GRIPPER_JOINT_NAMES
        target_joints.position = [close_position]
        success = self.gripper_commander.go(target_joints, wait=wait)
        self.gripper_commander.stop()
        self.gripper_commander.clear_pose_targets()
        return success

    def open_gripper(self):
        pass

    def spin(self):
        loop = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.yolo_model.predict_and_publish_result()
            loop.sleep()


def main():

    rospy.init_node("move_group_python_interface_tutorial", anonymous=False)

    # arm_controller = ArmController()

    initial_pose = np.array(
        [
            0.10570783048439313,
            -0.4794038027775048,
            0.8662147864558871,
            0.0009631401104561417,
            0.9880193696004536,
            -0.1540516258275005,
            0.009213806778614771,
        ]
    )

    # O BOZHE SKOLKO ZHE KOSTYLOV
    # poses = []
    # joint_states = []
    # num_examples = 50
    # for i in range(num_examples):
    #     rot_transform = R.from_euler("z", [i * 2 * np.pi / num_examples], degrees=False)
    #     target_pose = np.array(
    #         [
    #             *rot_transform.apply(initial_pose[:3])[0],
    #             *(rot_transform * R.from_quat(initial_pose[3:])).as_quat()[0],
    #         ]
    #     )
    #     arm_controller.move_trajectory([target_pose])
    #     joint_state = arm_controller.get_manipulator_joint_states()
    #     poses.append(target_pose)
    #     joint_states.append(joint_state)

    # correspondence_dict = {"poses": poses, "joint_states": joint_states}
    # np.save(
    #     os.path.join(rp.get_path("solution_master"), "correspondence/poses.npy"),
    #     correspondence_dict,
    # )

    # arm_controller.move_trajectory([initial_pose])
    # arm_controller = ArmController()

    # correspondences = np.load(
    #     rp.get_path("solution_master") + "/correspondence/poses.npy", allow_pickle=True
    # ).item()

    # # setting speed down
    # # arm_controller.manipulator_commander.set_max_velocity_scaling_factor(0.3)
    # for i in range(100):
    #     target_joinnt_states = correspondences["joint_states"][np.random.randint(0, 50)]
    #     arm_controller.move_joints(target_joinnt_states)

    while not rospy.is_shutdown():
        try:
            arm_controller = ArmController()
            arm_controller.move_to_home()
            arm_controller.init_environment_poses()
            while not rospy.is_shutdown():
                arm_controller.strategy()
        except Exception as e:
            print(e)

    # print(f"current pose: {arm_controller.get_current_grip_pose()}")
    # arm_controller.collect_scene_point_cloud()
    # for _ in range(100):
    #     arm_controller.move_to_inspect_point()
    #     arm_controller.detect_and_grip_object()
    #     arm_controller.move_to_inspect_point()
    # arm_controller.save_rgb_and_depth()
    # arm_controller.collect_dataset_samples()
    # trajectory_points = [
    #     [
    #         0.2648204544910632,
    #         0.00030229216542194737,
    #         0.9980174416225138,
    #         -0.7001528699810571,
    #         0.7124624113456665,
    #         -0.03097791104407552,
    #         0.03498056752187769,
    #     ],
    #     [
    #         0.5230977838755629,
    #         -0.01036933148596859,
    #         1.0335494720038685,
    #         -0.7174903308955812,
    #         0.6957961443776718,
    #         -0.013173797382636047,
    #         0.030030011705570847,
    #     ],
    #     [
    #         0.3,
    #         -0.0,
    #         1.1,
    #         -0.6447705311538084,
    #         0.7165366385001714,
    #         -0.183031483277461,
    #         0.19325031428632358,
    #     ],
    # ]

    # for i in range(len(trajectory_points)):
    #     trajectory_points[i] = np.array(trajectory_points[i])
    # arm_controller.move_trajectory(trajectory_points)
    # pass
    # arm_controller.collect_scene_point_cloud()

    # arm_controller.move_to_home()
    # arm_controller.collect_dataset_samples()
    # arm_controller.move_to_home()
    # arm_controller.spin()
    # arm_controller.yolo_model.predict_and_publish_result()


if __name__ == "__main__":
    main()
