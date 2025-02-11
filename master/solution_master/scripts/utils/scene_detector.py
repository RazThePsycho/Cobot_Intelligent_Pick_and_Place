import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


class Mask:
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        self.lower = lower
        self.upper = upper
        self.lower_float = self.lower / 255
        self.upper_float = self.upper / 255


class SceneDetector:
    def __init__(self) -> None:
        self.blue_mask = Mask(
            np.array(
                [
                    1,
                    15,
                    17,
                ]
            ),
            np.array([15, 128, 127]),
        )
        self.orange_mask = Mask(np.array([10, 5, 0]), np.array([110, 60, 10]))
        self.z_min = 0.4
        self.x_min = 0.3
        self.y_min = 0.3
        self.place_z_offset = 0.1
        self.place_xy_offset = 0.05
        self.table_offset = 0.15
        self.table_z = 0.5

    def get_place_se3(self, bounding_box):
        center = np.array(bounding_box.get_center())
        max_z = np.array(bounding_box.get_max_bound())[2]
        center[2] = max_z + self.place_z_offset
        if center[2] < 1.0:
            # TODO: DELETE THIS TRASH
            center[2] = 1.0

        x, y, _ = center

        # Z-axis (normalized) in the global frame, projected onto the XY plane
        z_axis = np.array([x, y, 0])
        z_axis_norm = z_axis / np.linalg.norm(z_axis)

        # Y-axis is parallel to the global Z-axis
        y_axis = -np.array([0, 0, 1])

        # X-axis, found by cross product of Y and Z axes
        x_axis = np.cross(y_axis, z_axis_norm)

        # Create transformation matrix with X, Y, Z as columns
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = np.column_stack((x_axis, y_axis, z_axis_norm))
        transformation_matrix[:3, 3] = center
        transformation_matrix[0, 3] = (
            transformation_matrix[0, 3]
            - np.sign(transformation_matrix[0, 3]) * self.place_xy_offset
        )
        transformation_matrix[1, 3] = (
            transformation_matrix[1, 3]
            - np.sign(transformation_matrix[1, 3]) * self.place_xy_offset
        )
        return transformation_matrix

    @staticmethod
    def get_point_cloud_by_mask(src_point_cloud: o3d.geometry.PointCloud, mask: Mask):
        colors = np.asarray(src_point_cloud.colors)
        lower_bound = mask.lower_float
        upper_bound = mask.upper_float
        mask = (
            (colors[:, 0] > lower_bound[0])
            & (colors[:, 0] < upper_bound[0])
            & (colors[:, 1] > lower_bound[1])
            & (colors[:, 1] < upper_bound[1])
            & (colors[:, 2] > lower_bound[2])
            & (colors[:, 2] < upper_bound[2])
        )
        filtered_points = np.asarray(src_point_cloud.points)[mask]
        filtered_colors = np.asarray(src_point_cloud.colors)[mask]
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        return filtered_pcd

    def compute_table_so3(self, table_pcd, table_bounding_box):
        table_center = np.array(table_bounding_box.get_center())
        RZ = R.from_euler(
            "z", np.arctan2(table_center[1], table_center[0]), degrees=False
        )
        points_array = np.asarray(table_pcd.points)
        xy_norm = np.linalg.norm(points_array[:, :2], axis=1)
        distance_to_table = (
            np.mean(np.sort(xy_norm)[:30]).mean().item() + self.table_offset
        )
        table_center = RZ.apply(np.array([distance_to_table, 0, self.table_z]))
        table_transform = np.eye(4)
        table_transform[:3, :3] = RZ.as_matrix()
        table_transform[:3, 3] = table_center
        return table_transform

    def compute_trashbox_bounding_box(
        self, src_point_cloud: o3d.geometry.PointCloud, mask: Mask
    ):
        filtered_pcd = self.get_point_cloud_by_mask(src_point_cloud, mask)
        # Cluster the point cloud
        labels = np.array(
            filtered_pcd.cluster_dbscan(eps=0.05, min_points=50, print_progress=True)
        )
        max_label = labels.max()
        largest_cluster_label = max(
            range(max_label + 1), key=lambda x: list(labels).count(x)
        )
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

        # Extract the largest cluster
        largest_cluster = filtered_pcd.select_by_index(largest_cluster_indices)

        largets_aobb = largest_cluster.get_axis_aligned_bounding_box()
        largets_aobb.color = (1, 0, 0)
        return largets_aobb

    def delete_bounding_box_from_pcd(
        self,
        src_point_cloud: o3d.geometry.PointCloud,
        bounding_box: o3d.geometry.AxisAlignedBoundingBox,
    ):
        points = np.asarray(src_point_cloud.points)
        min_bound = np.array(bounding_box.get_min_bound())
        max_bound = np.array(bounding_box.get_max_bound())
        outside_indices = np.where(
            (points[:, 0] < min_bound[0])
            | (points[:, 0] > max_bound[0])
            | (points[:, 1] < min_bound[1])
            | (points[:, 1] > max_bound[1])
            | (points[:, 2] < min_bound[2])
            | (points[:, 2] > max_bound[2])
        )[0]
        filtered_pcd = src_point_cloud.select_by_index(outside_indices)
        return filtered_pcd

    def compute_scene_objects_poses(self, pcd_list: list):
        res_pcd = o3d.geometry.PointCloud()
        for i in range(len(pcd_list)):
            # Convert the point cloud to a NumPy array
            points = np.asarray(pcd_list[i].points)

            # Correctly use np.where without additional brackets
            indices_z = np.where((points[:, 2] >= self.z_min))[0]
            indices_xy = np.where(
                (points[:, 0] >= self.x_min)
                | (points[:, 1] >= self.y_min)
                | (points[:, 0] <= -self.x_min)
                | (points[:, 1] <= -self.y_min)
            )[0]
            indices = np.intersect1d(indices_z, indices_xy)

            # Select points based on indices and update the point cloud in the list
            pcd_list[i] = pcd_list[i].select_by_index(indices)

            # Accumulate the selected points into res_pcd
            res_pcd += pcd_list[i]
        res_pcd = res_pcd.voxel_down_sample(voxel_size=0.01)
        blue_trashbox_bounding_box = self.compute_trashbox_bounding_box(
            res_pcd, self.blue_mask
        )
        orange_trashbox_bounding_box = self.compute_trashbox_bounding_box(
            res_pcd, self.orange_mask
        )
        res_pcd = self.delete_bounding_box_from_pcd(res_pcd, blue_trashbox_bounding_box)
        res_pcd = self.delete_bounding_box_from_pcd(
            res_pcd, orange_trashbox_bounding_box
        )
        labels = np.array(
            res_pcd.cluster_dbscan(eps=0.05, min_points=50, print_progress=True)
        )
        max_label = labels.max()
        largest_cluster_label = max(
            range(max_label + 1), key=lambda x: list(labels).count(x)
        )
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

        # Extract the largest cluster
        largest_cluster = res_pcd.select_by_index(largest_cluster_indices)
        table_aobb = largest_cluster.get_oriented_bounding_box()
        table_aobb.color = (0, 1, 0)

        blue_trashbox_se3 = self.get_place_se3(blue_trashbox_bounding_box)
        orange_trashbox_se3 = self.get_place_se3(orange_trashbox_bounding_box)
        table_se3 = self.compute_table_so3(largest_cluster, table_aobb)
        plane_model, inliers = largest_cluster.segment_plane(
            distance_threshold=0.0001, ransac_n=3, num_iterations=3000
        )

        return {
            "blue_trashbox_se3": blue_trashbox_se3,
            "orange_trashbox_se3": orange_trashbox_se3,
            "table_se3": table_se3,
            "plane_coefficients": plane_model,
        }
