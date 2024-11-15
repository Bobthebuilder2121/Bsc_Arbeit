from copy import deepcopy
from datetime import datetime
import open3d as o3d
import collections
import struct
import argparse
import numpy as np
import os

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length,)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def read_point3d_binary(path_to_model_file):
    points3D = []
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            pt_id = binary_point_line_properties[0]
            XYZ = np.array(binary_point_line_properties[1:4])
            RGB = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length,)
            candidate_ids = track_elems[0::2]
            points3D.append({
                "id": pt_id,
                "XYZ": XYZ,
                "RGB": RGB,
                "error": error,
                "candidate_ids": candidate_ids
            })

    XYZs = np.array([point["XYZ"] for point in points3D])
    RGBs = np.array([point["RGB"] for point in points3D]) / 255.0
    candidate_ids = {point["id"]: point["candidate_ids"] for point in points3D}

    return XYZs, RGBs, candidate_ids

def save_point_cloud(input_path, output_path=None):
    """
    Saves the point cloud data as a PCD file. Checks if the output directory exists
    and creates a warning if it does not.

    Parameters:
    input_path (str): The path to the input binary file (used to read the point cloud data).
    output_path (str, optional): The path to save the PCD file. If None, saves as 'points3D.pcd' in the current directory.
    """
    
    # Read point cloud data from the binary file
    xyz, colors_rgb, candidate_ids = read_point3d_binary(input_path)

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

    # If output_path is not provided, save as 'points3D.pcd' in the current directory
    if output_path is None:
        output_path = "points3D.pcd"
    else:
        # Ensure the output path has the correct extension
        if output_path.endswith(".bin"):
            output_path = output_path.replace(".bin", ".pcd")
            
        if not output_path.endswith(".pcd"):
            output_path += ".pcd"

        # Check if the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            print(f"Warning: The directory '{output_dir}' does not exist!")
            return
    
    # Save the PointCloud object to a PCD file

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to: {output_path}")
    print(f"Point cloud data shape: {xyz.shape}, {colors_rgb.shape}")

