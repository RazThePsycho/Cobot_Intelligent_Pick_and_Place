import numpy as np


def create_inspection_so3_in_table_frame(point, target):
    # Convert points to numpy arrays for ease of computation
    point = np.array(point)
    target = np.array(target)

    # Calculate the Z-axis direction (unit vector)
    z_axis = target - point
    z_axis = z_axis / np.linalg.norm(z_axis)

    # Calculate the X-axis direction
    # Use global Y-axis to generate an orthogonal vector in the XZ plane
    global_y = np.array([-1, 0, 0])
    x_axis = np.cross(global_y, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Calculate the Y-axis direction
    y_axis = np.cross(z_axis, x_axis)

    # Create the rotation matrix using the X, Y, and Z axes
    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Create the translation vector
    translation_vector = point

    # Combine into a 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector

    return transform_matrix


def create_inspection_so3_in_global_frame(point, target, table_transform):
    return table_transform @ create_inspection_so3_in_table_frame(point, target)


def calculate_so3_oriented_to_target(origin: np.ndarray, target: np.ndarray):
    """calcualates so3 matrix to orient the frame to look at the target point, the y axis is directed to the global origin

    Args:
        origin (np.ndarray): coordinates of the origin of the frame
        target (np.ndarray):
    """

    def normalize(v):
        return v / np.linalg.norm(v)

    # Calculate Z-axis
    z_axis = normalize(target - origin)

    # Explicitly set the Y-axis to be parallel to the world X-axis
    y_axis = np.array(
        [-1, 0, 0]
    )  # This ensures Y-axis projection on OXY is parallel to the world X-axis

    # Calculate the X-axis using cross product to ensure orthogonality and right-handed coordinate system
    # Since Z-axis might be parallel or almost parallel to Y-axis, we need a conditional check to prevent invalid cross product
    if np.allclose(z_axis, y_axis) or np.allclose(z_axis, -y_axis):
        # Z-axis is parallel or anti-parallel to Y-axis, choose a different approach
        x_axis = np.array(
            [0, 0, 1]
        )  # Default X-axis if Z is parallel/anti-parallel to Y
    else:
        x_axis = normalize(np.cross(y_axis, z_axis))
        # Recalculate Y-axis to ensure orthogonality, as the initial Y might not be perfectly orthogonal due to numerical errors
        y_axis = np.cross(z_axis, x_axis)

    se3_matrix = np.eye(4)
    # Construct the SE3 matrix
    se3_matrix[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    se3_matrix[:3, 3] = origin
    return se3_matrix


def generate_ellipse_trajectory(
    target_point,
    num_points=20,
    x_center=0.8,
    y_center=0.0,
    radius_x=0.2,
    radius_y=0.3,
    z_coordinate=1,
):
    origins = [
        np.array(
            [
                x_center + radius_x * 2 * np.sin(t),
                y_center + radius_y * np.cos(t),
                z_coordinate,
            ]
        )
        for t in np.linspace(-np.pi / 4, 5 * np.pi / 4, num_points)
    ]
    return [
        calculate_so3_oriented_to_target(origin, target_point) for origin in origins
    ]


def point_line_distance(x1, y1, x2, y2, x0=0, y0=0):
    """
    Calculate the distance from the origin to the line defined by two points (x1, y1) and (x2, y2).
    Also checks if the closest point is within the segment.
    """
    # Line segment vector
    dx, dy = x2 - x1, y2 - y1
    # Vector from point 1 to the origin
    dx0, dy0 = x0 - x1, y0 - y1
    # Project vector from point 1 to origin onto the line segment vector
    t = (dx * dx0 + dy * dy0) / (dx**2 + dy**2)
    # Find the closest point on the line segment to the origin
    closest_x, closest_y = x1 + t * dx, y1 + t * dy
    # Check if the closest point is within the segment
    within_segment = 0 <= t <= 1
    # Distance from the origin to the closest point
    distance = np.sqrt((closest_x - x0) ** 2 + (closest_y - y0) ** 2)
    return distance, within_segment, closest_x, closest_y


def calculate_angle(x1, y1, x2, y2):
    """
    Calculate the angle (in degrees) between the line segment connecting (x1, y1) and (x2, y2) and the x-axis.
    """
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angle


def check_distance_and_angle(x1, y1, x2, y2, threshold):
    distance, within_segment, _, _ = point_line_distance(x1, y1, x2, y2)
    if distance < threshold and within_segment:
        angle = calculate_angle(x1, y1, x2, y2)
        return True, angle
    else:
        return False, None
