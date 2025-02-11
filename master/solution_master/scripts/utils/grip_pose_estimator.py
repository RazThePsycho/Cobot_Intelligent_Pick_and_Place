import os
import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from rospkg import RosPack
from utils.utils import calculate_so3_oriented_to_target

rp = RosPack()


def convert_se3_to_table_frame(table_pose, object_pose):
    table_pose_inv = np.eye(4)
    table_pose_inv[:3, :3] = table_pose[:3, :3].T
    table_pose_inv[:3, 3] = -table_pose[:3, :3].T @ table_pose[:3, 3]
    return table_pose_inv @ object_pose


class GripPoseEstimator:
    def __init__(
        self,
        trashcan_mesh_path=os.path.join(
            rp.get_path("solution_master"), "models/trashcan.stl"
        ),
        tetrapack_mesh_path=os.path.join(
            rp.get_path("solution_master"), "models/tetrapack.stl"
        ),
    ) -> None:
        self.trashcan_pcd = self.get_trashcan_pcd(trashcan_mesh_path)
        self.tetrapack_pcd = self.get_tetraback_pcd(tetrapack_mesh_path)
        self.pick_offset = 0.1
        self.table_pose = None
        self.table_angle = -7
        self.min_z = 0.0
        self.min_x = -0.03
        self.max_y = 0.3
        self.threshold = 0.05
        self.vector_scale = 0.08

    @staticmethod
    def get_trashcan_pcd(trashcan_path):
        mesh = o3d.io.read_triangle_mesh(trashcan_path)
        mesh.rotate(
            R.from_euler("z", -90, degrees=True).as_matrix(), center=mesh.get_center()
        )
        centroid = mesh.get_center()
        # Translate the mesh to center it at the origin
        mesh.translate(-centroid)
        pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
        return pcd

    @staticmethod
    def get_tetraback_pcd(tetrapack_path):
        mesh = o3d.io.read_triangle_mesh(tetrapack_path)
        mesh.rotate(
            R.from_euler("z", -90, degrees=True).as_matrix(), center=mesh.get_center()
        )
        centroid = mesh.get_center()
        # Translate the mesh to center it at the origin
        mesh.translate(-centroid)
        mesh.scale(2.0, center=mesh.get_center())
        pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
        return pcd

    def check_grip_so3(self, grip_so3):
        if grip_so3[2, 3] < self.min_z:
            return 0
        if (
            grip_so3[0, 3] > self.min_x
            and grip_so3[1, 3] > -self.max_y
            and grip_so3[1, 3] < self.max_y
        ):
            return 1
        return 2

    def fix_grip_poses(self, grip_poses, table_pose):
        grip_poses = [
            convert_se3_to_table_frame(table_pose, grip_pose)
            for grip_pose in grip_poses
        ]
        print("GRIP POSES", grip_poses)
        check_result = self.check_grip_so3(grip_poses[1])
        if check_result == 0:
            print("Grip pose is too LOW, ABORTING")
            return None
        if check_result == 1:
            print("Grip pose is GOOD")
            return [self.table_pose @ grip_pose for grip_pose in grip_poses]
        print("FIXING GRIP POSE")
        additional_vector = np.array([0, 0, self.pick_offset * 1.4])
        target_additional_vector = np.array([0, 0, 0])
        if grip_poses[1][0, 3] < self.min_x:
            additional_vector[0] = self.vector_scale
            target_additional_vector[0] = self.vector_scale / 5
        if grip_poses[1][1, 3] < -self.max_y:
            additional_vector[1] = self.vector_scale
            target_additional_vector[1] = self.vector_scale / 5
        if grip_poses[1][1, 3] > self.max_y:
            additional_vector[1] = -self.vector_scale
            target_additional_vector[1] = -self.vector_scale / 5
        print("ADDITIONAL VECTOR", additional_vector)
        point = grip_poses[1][:3, 3] + additional_vector
        target = grip_poses[1][:3, 3]
        new_pre_grip_so3 = calculate_so3_oriented_to_target(point, target)
        print("NEW PRE GRIP SO3", new_pre_grip_so3)
        new_grip_so3 = np.eye(4)
        new_grip_so3[:3, :3] = new_pre_grip_so3[:3, :3]
        new_grip_so3[:3, 3] = grip_poses[1][:3, 3]
        print("NEW GRIP SO3", new_grip_so3)
        print("NEW PRE GRIP SO3", new_pre_grip_so3)
        # transforming to table frame
        new_grip_so3 = self.table_pose @ new_grip_so3
        new_pre_grip_so3 = self.table_pose @ new_pre_grip_so3
        return [new_pre_grip_so3, new_grip_so3]

    @staticmethod
    def align_gripper_y_to_target_x(gripper_se3, target_se3):
        """
        Rotate the gripper SE(3) matrix so that its y-axis projection onto the OXY plane
        aligns with the target SE(3) matrix's x-axis projection.

        Parameters:
        gripper_se3 (np.array): The SE(3) matrix of the gripper.
        target_se3 (np.array): The SE(3) matrix of the target.

        Returns:
        np.array: The rotated gripper SE(3) matrix.
        """
        # Extract the y-axis vector of the gripper and x-axis vector of the target
        y_gripper = gripper_se3[:3, 1]
        x_target = target_se3[:3, 0]

        # Project onto the OXY plane (make Z component zero)
        y_gripper[2] = 0
        x_target[2] = 0

        # Normalize the projections to ensure they are pure directions
        y_gripper_normalized = y_gripper / np.linalg.norm(y_gripper)
        x_target_normalized = x_target / np.linalg.norm(x_target)

        # Calculate the angle between the two vectors
        dot_product = np.dot(y_gripper_normalized, x_target_normalized)
        angle = np.arccos(
            np.clip(dot_product, -1.0, 1.0)
        )  # Clip to handle numerical errors

        # Determine the direction of rotation using cross product (sign of Z component)
        cross_product = np.cross(y_gripper_normalized, x_target_normalized)
        if cross_product[2] < 0:
            angle = -angle

        # Create a rotation matrix around the Z-axis
        R_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Apply the rotation to the gripper's SE(3) matrix
        rotated_gripper_se3 = np.eye(4)
        rotated_gripper_se3[:3, :3] = np.dot(R_z[:3, :3], gripper_se3[:3, :3])
        rotated_gripper_se3[:3, 3] = gripper_se3[:3, 3]

        return rotated_gripper_se3

    @staticmethod
    def align_x_axes(gripper_se3, target_se3):
        """
        Rotate the gripper SE(3) matrix so that its x-axis projection onto the OXY plane
        aligns with the target SE(3) matrix's x-axis projection.

        Parameters:
        gripper_se3 (np.array): The SE(3) matrix of the gripper.
        target_se3 (np.array): The SE(3) matrix of the target.

        Returns:
        np.array: The rotated gripper SE(3) matrix.
        """
        # Extract the x-axis vectors
        x_gripper = gripper_se3[:3, 0]
        x_target = target_se3[:3, 0]

        # Project onto the OXY plane (make Z component zero)
        x_gripper[2] = 0
        x_target[2] = 0

        # Normalize the projections to ensure they are pure directions
        x_gripper_normalized = x_gripper / np.linalg.norm(x_gripper)
        x_target_normalized = x_target / np.linalg.norm(x_target)

        # Calculate the angle between the two vectors
        dot_product = np.dot(x_gripper_normalized, x_target_normalized)
        angle = np.arccos(
            np.clip(dot_product, -1.0, 1.0)
        )  # Clip to handle numerical errors gracefully

        # Determine the direction of rotation using cross product (sign of Z component)
        cross_product = np.cross(x_gripper_normalized, x_target_normalized)
        if cross_product[2] < 0:
            angle = -angle

        # Create a rotation matrix around the Z-axis
        R_z = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Apply the rotation to the gripper's SE(3) matrix
        # rotated_gripper_se3 = np.dot(R_z, gripper_se3)
        rotated_gripper_se3 = np.eye(4)
        rotated_gripper_se3[:3, :3] = np.dot(R_z[:3, :3], gripper_se3[:3, :3])
        rotated_gripper_se3[:3, 3] = gripper_se3[:3, 3]

        return rotated_gripper_se3

    def generate_grasp_trajectory(self, target_se3, predicted_class):
        pick_offset_projection = (
            R.from_matrix(self.table_pose[:3, :3])
            * R.from_euler("y", self.table_angle, degrees=True)
        ).apply(np.array([0, 0, self.pick_offset]))
        gripper_se3 = calculate_so3_oriented_to_target(
            target_se3[:3, 3] + pick_offset_projection, target_se3[:3, 3]
        )
        gripper_se3 = self.align_gripper_y_to_target_x(gripper_se3, target_se3)
        # TODO: FIX THIS rotating on 180 degrees in local frame
        gripper_se3[:3, :3] = (
            gripper_se3[:3, :3] @ R.from_euler("z", 180, degrees=True).as_matrix()
        )
        pre_grasp_se3 = np.copy(gripper_se3)
        grasp_se3 = gripper_se3
        grasp_se3[:3, 3] = grasp_se3[:3, 3] - pick_offset_projection
        if predicted_class == 0:
            print(f"{'-' * 10} Normalizing {'-' * 10}")

            grasp_se3[:3, :3] = (
                self.table_pose[:3, :3]
                @ R.from_euler("y", self.table_angle, degrees=True).as_matrix()
                @ grasp_se3[:3, :3]
            )
            pre_grasp_se3[:3, :3] = grasp_se3[:3, :3]

        print(f"{'-' * 10} IN TABLE COORDS: {'-' * 10}")
        print(
            "PRE GRASP",
            convert_se3_to_table_frame(self.table_pose, pre_grasp_se3),
        )
        print("GRASP", convert_se3_to_table_frame(self.table_pose, grasp_se3))

        checked_grasp = self.fix_grip_poses([pre_grasp_se3, grasp_se3], self.table_pose)
        return checked_grasp

    def icp_pose_estimation(self, source, target):
        """_summary_

        Args:
            source (_type_): source mesh point cloud
            target (_type_): estimated point cloud
        """

        source = copy.deepcopy(source)
        # Initial alignment with bounding box
        aobb = target.get_oriented_bounding_box()
        T = np.eye(4)
        T[:3, :3] = aobb.R
        T[:3, 3] = aobb.center
        threshold_schedule = [4, 2, 0.5, 0.1, 0.01]
        source.transform(T)

        # print(T)
        # for threshold in threshold_schedule:
        #     # print(f"Apply point-to-point ICP with treshold {threshold}")
        #     reg_p2p = o3d.pipelines.registration.registration_icp(
        #     source,target, threshold,np.eye(4),
        #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #     )
        #     T = np.dot(reg_p2p.transformation,T)
        #     source.transform(reg_p2p.transformation)
        # print(reg_p2p)

        return T, source

    def calculate_grip_poses(self, yolo_predictions, rgb_image, depth_image, metadata):
        detected_pcds = []
        masked_pcds = []
        intrinsic_matrix = metadata["camera_intrinsic"]
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640,
            height=480,
            fx=intrinsic_matrix[0, 0],
            fy=intrinsic_matrix[1, 1],
            cx=intrinsic_matrix[0, 2],
            cy=intrinsic_matrix[1, 2],
        )
        if len(yolo_predictions) == 0:
            return None
        predicted_classes = []
        for pred in yolo_predictions:
            predicted_classes.append(pred.boxes.cls.cpu().numpy().astype(int))
            depth_masked = np.copy(depth_image)
            depth_masked[(pred.masks.data[0].cpu().numpy() < 0.1)] = 0
            o3d_rgb_image = o3d.geometry.Image(rgb_image)
            o3d_depth_image = o3d.geometry.Image(depth_masked)
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
            masked_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, o3d_camera_intrinsic
            )
            masked_pcd = masked_pcd.transform(camera_transform)
            masked_pcds.append(masked_pcd)

        for pcd in masked_pcds:
            to_vis_pcd = pcd
            # Parameters for outlier removal
            nb_neighbors = 40  # Number of neighbors to consider
            std_ratio = 0.1  # Standard deviation ratio

            # Apply StatisticalOutlierRemoval filter
            filtered_cloud, _ = to_vis_pcd.remove_statistical_outlier(
                nb_neighbors, std_ratio
            )
            # o3d.visualization.draw_geometries([to_vis_pcd])
            aobb = filtered_cloud.get_oriented_bounding_box()
            aobb.color = [1, 0, 0]
            # Visualize the result
            # o3d.visualization.draw_geometries([filtered_cloud,aobb])
        # o3d.visualization.draw_geometries([to_vis_pcd])
        objects_poses = []
        for masked_pcd in masked_pcds:
            point_cloud = copy.deepcopy(self.trashcan_pcd)
            T, aligned_point_cloud = self.icp_pose_estimation(point_cloud, masked_pcd)
            objects_poses.append(T)

        # obtaining grasp trajectory

        grasp_trajectory = self.generate_grasp_trajectory(
            objects_poses[0], predicted_classes[0]
        )

        return grasp_trajectory, predicted_classes
