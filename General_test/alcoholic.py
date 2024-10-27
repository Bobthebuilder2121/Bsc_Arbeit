import open3d as o3d
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_ply_as_pointcloud(file_path, num_points=10000000):
    """
    Load a .ply mesh file as a point cloud.
    Sample points uniformly based on the provided num_points.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Sample points uniformly, ensuring both meshes will have the same number of points
    point_cloud = mesh.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(point_cloud.points)

def match_point_clouds(pcd1, pcd2):
    """
    Match points from pcd2 to pcd1 using nearest neighbor search.
    This ensures that for each point in pcd1, we find the closest point in pcd2.
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pcd2)
    distances, indices = nbrs.kneighbors(pcd1)
    
    # Reorganize pcd2 based on nearest neighbors to pcd1
    matched_pcd2 = pcd2[indices.flatten()]
    
    return matched_pcd2

def compute_psnr(pcd1, pcd2, max_value=1.0):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two point clouds.
    Arguments:
    - pcd1, pcd2: Point clouds as Nx3 numpy arrays.
    - max_value: Maximum possible value for the points (default 1.0, adjust if necessary).
    """
    # Ensure both point clouds have the same number of points
    if pcd1.shape != pcd2.shape:
        raise ValueError("The point clouds must have the same shape for comparison")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert point clouds to tensors
    pcd1_torch = torch.tensor(pcd1, dtype=torch.float32).to(device)
    pcd2_torch = torch.tensor(pcd2, dtype=torch.float32).to(device)
    
    # Compute Mean Squared Error (MSE)
    mse = torch.mean((pcd1_torch - pcd2_torch) ** 2)
    
    if mse == 0:
        return float('inf')  # Infinite PSNR when MSE is zero

    # Compute PSNR
    psnr = 20 * torch.log10(torch.tensor(max_value)) - 10 * torch.log10(mse)
    
    return psnr.item()

def compute_ssim(pcd1, pcd2, C1=0.01**2, C2=0.03**2):
    """
    Compute the SSIM (Structural Similarity Index) between two point clouds.
    Arguments:
    - pcd1, pcd2: Point clouds as Nx3 numpy arrays.
    - C1, C2: Constants to stabilize the division when the denominator is small.
    """
    # Ensure both point clouds have the same number of points
    if pcd1.shape != pcd2.shape:
        raise ValueError("The point clouds must have the same shape for comparison")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert point clouds to tensors
    pcd1_torch = torch.tensor(pcd1, dtype=torch.float32).to(device)
    pcd2_torch = torch.tensor(pcd2, dtype=torch.float32).to(device)

    # Compute mean
    mean_pcd1 = torch.mean(pcd1_torch)
    mean_pcd2 = torch.mean(pcd2_torch)

    # Compute variance
    variance_pcd1 = torch.var(pcd1_torch)
    variance_pcd2 = torch.var(pcd2_torch)

    # Compute covariance
    covariance = torch.mean((pcd1_torch - mean_pcd1) * (pcd2_torch - mean_pcd2))

    # Compute SSIM
    ssim = ((2 * mean_pcd1 * mean_pcd2 + C1) * (2 * covariance + C2)) / \
           ((mean_pcd1 ** 2 + mean_pcd2 ** 2 + C1) * (variance_pcd1 + variance_pcd2 + C2))
    
    return ssim.item()

def main():
    # Load the two .ply files
    file1 = "./gtcat.ply"
    file2 = "./redone_cat.ply"
    
    pcd1 = load_ply_as_pointcloud(file1)
    pcd2 = load_ply_as_pointcloud(file2)

    # Match pcd2 points to pcd1 using nearest neighbor search
    matched_pcd2 = match_point_clouds(pcd1, pcd2)

    # Compute PSNR
    psnr_value = compute_psnr(pcd1, matched_pcd2)
    print(f"PSNR between the two point clouds: {psnr_value:.2f} dB")
    
    # Compute SSIM
    ssim_value = compute_ssim(pcd1, matched_pcd2)
    print(f"SSIM between the two point clouds: {ssim_value:.4f}")

if __name__ == "__main__":
    main()