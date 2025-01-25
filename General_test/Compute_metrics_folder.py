import open3d as o3d
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mesh2sdf import compute  # Import mesh2sdf for SDF-based calculations
import point_cloud_utils as pcu
import os
import csv

def load_ply_as_pointcloud(file_path, num_points=2000000):
    """
    Load a .ply mesh file as a point cloud.
    Sample points uniformly based on the provided num_points.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(point_cloud.points), mesh

def match_point_clouds(pcd1, pcd2):
    """
    Match points from pcd2 to pcd1 using nearest neighbor search.
    Ground truth (pcd1) is treated as more exact, so pcd2 points are reordered
    to match the ground truth's sampling.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pcd2)
    distances, indices = nbrs.kneighbors(pcd1)
    
    # Reorganize pcd2 based on nearest neighbors to pcd1
    matched_pcd2 = pcd2[indices.flatten()]
    return matched_pcd2

def compute_iou_with_sdf(mesh1, mesh2, resolution=1000):
    """
    Compute IoU (Intersection over Union) between two meshes using Signed Distance Field (SDF).
    
    Args:
        mesh1, mesh2: Open3D TriangleMesh objects.
        resolution: Resolution of the SDF grid.

    Returns:
        IoU value (float).
    """
    vertices1 = np.asarray(mesh1.vertices)
    triangles1 = np.asarray(mesh1.triangles)
    
    vertices2 = np.asarray(mesh2.vertices)
    triangles2 = np.asarray(mesh2.triangles)

    sdf1 = compute(vertices1, triangles1, resolution, fix=False, return_mesh=False)
    sdf2 = compute(vertices2, triangles2, resolution, fix=False, return_mesh=False)
    vol1 = sdf1 < 0  # Inside the first mesh
    vol2 = sdf2 < 0  # Inside the second mesh

    intersection = np.sum(vol1 & vol2)
    union = np.sum(vol1 | vol2)
    
    if union == 0:
        return 0.0

    return intersection / union

def main():
    groundtruth_file = "/workspace/data/data_reconstruction/cat_benchmarks/gtcat.ply"
    base_dir = "/workspace/data/data_reconstruction/cat_benchmarks/top_view/"
    output_csv = "evaluation_results.csv"

    # Load ground truth mesh
    _, mesh_gt = load_ply_as_pointcloud(groundtruth_file)

    results = []

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Locate the aligned .ply file
        aligned_file = os.path.join(folder_path, "Outputs/sugar/", f"alligned_{folder_name}.ply")
        print(f"Checking folder: {aligned_file}")
        if not os.path.exists(aligned_file):
            print(f"Aligned file not found for folder: {folder_name}")
            continue

        # Load predicted point cloud and mesh
        pcd_pr, mesh_pr = load_ply_as_pointcloud(aligned_file)

        # Load vertices for Chamfer distance calculation
        p1 = pcu.load_mesh_v(groundtruth_file)
        p2 = pcu.load_mesh_v(aligned_file)

        # Compute Chamfer distance
        chamfer_distance = pcu.chamfer_distance(p1, p2)

        # Compute IoU using mesh2sdf
        iou_value = compute_iou_with_sdf(mesh_gt, mesh_pr, resolution=64)

         # Store results
        results.append([folder_name, f"{chamfer_distance:.6f}".replace('.', ','), f"{iou_value:.8f}".replace('.', ',')])
        print(f"Folder: {folder_name}, Chamfer Distance: {chamfer_distance:.6f}, IoU: {iou_value:.8f}")

    # Print all results
    print("\nSummary of Results:")
    for result in results:
        print(f"Folder: {result[0]}, Chamfer Distance: {result[1]}, IoU: {result[2]}")

    # Save results to CSV
    try:
        with open(output_csv, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Folder", "Chamfer Distance", "IoU"])
            writer.writerows(results)
        print(f"\nResults saved to {output_csv}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

if __name__ == "__main__":
    main()
