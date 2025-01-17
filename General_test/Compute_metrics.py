import open3d as o3d
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from mesh2sdf import compute  # Import mesh2sdf for SDF-based calculations

def load_ply_as_pointcloud(file_path, num_points=1000000):
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

def compute_psnr(pcd1, pcd2, max_value=1.0):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two point clouds.
    """
    if pcd1.shape != pcd2.shape:
        raise ValueError("The point clouds must have the same shape for comparison")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcd1_torch = torch.tensor(pcd1, dtype=torch.float32).to(device)
    pcd2_torch = torch.tensor(pcd2, dtype=torch.float32).to(device)

    mse = torch.mean((pcd1_torch - pcd2_torch) ** 2)
    if mse == 0:
        return float('inf')  # Infinite PSNR when MSE is zero

    psnr = 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
    return psnr.item()

def compute_ssim(pcd1, pcd2, C1=0.01**2, C2=0.03**2):
    """
    Compute the SSIM (Structural Similarity Index) between two point clouds.
    """
    if pcd1.shape != pcd2.shape:
        raise ValueError("The point clouds must have the same shape for comparison")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcd1_torch = torch.tensor(pcd1, dtype=torch.float32).to(device)
    pcd2_torch = torch.tensor(pcd2, dtype=torch.float32).to(device)

    mean_pcd1 = torch.mean(pcd1_torch)
    mean_pcd2 = torch.mean(pcd2_torch)

    variance_pcd1 = torch.var(pcd1_torch)
    variance_pcd2 = torch.var(pcd2_torch)

    covariance = torch.mean((pcd1_torch - mean_pcd1) * (pcd2_torch - mean_pcd2))

    ssim = ((2 * mean_pcd1 * mean_pcd2 + C1) * (2 * covariance + C2)) / \
           ((mean_pcd1 ** 2 + mean_pcd2 ** 2 + C1) * (variance_pcd1 + variance_pcd2 + C2))
    
    return ssim.item()

def compute_chamfer_distance(pcd1, pcd2):
    """
    Compute the Chamfer Distance between two point clouds.
    """
    nbrs_pcd2 = NearestNeighbors(n_neighbors=1).fit(pcd2)
    distances_pcd1_to_pcd2, _ = nbrs_pcd2.kneighbors(pcd1)

    nbrs_pcd1 = NearestNeighbors(n_neighbors=1).fit(pcd1)
    distances_pcd2_to_pcd1, _ = nbrs_pcd1.kneighbors(pcd2)

    chamfer_distance = (np.mean(distances_pcd1_to_pcd2**2) + np.mean(distances_pcd2_to_pcd1**2)) / 2
    return chamfer_distance

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
    benchmark_file = "/workspace/data/data_reconstruction/cat_benchmarks/rotated_cat.ply"

    # Load point clouds and meshes
    pcd_gt, mesh_gt = load_ply_as_pointcloud(groundtruth_file)
    pcd_pr, mesh_pr = load_ply_as_pointcloud(benchmark_file)

    # Match point clouds
    matched_pcd_pr = match_point_clouds(pcd_gt, pcd_pr)

    # Compute PSNR
    psnr_value = compute_psnr(pcd_gt, matched_pcd_pr)
    print(f"PSNR: {psnr_value:.2f} dB")

    # Compute SSIM
    ssim_value = compute_ssim(pcd_gt, matched_pcd_pr)
    print(f"SSIM: {ssim_value:.4f}")

    # Compute Chamfer Distance
    chamfer_dist = compute_chamfer_distance(pcd_gt, pcd_pr)
    print(f"Chamfer Distance: {chamfer_dist:.6f}")

    # Compute IoU using mesh2sdf
    iou_value = compute_iou_with_sdf(mesh_gt, mesh_pr, resolution=64)
    print(f"IoU (SDF-based): {iou_value:.6f}")

if __name__ == "__main__":
    main()
