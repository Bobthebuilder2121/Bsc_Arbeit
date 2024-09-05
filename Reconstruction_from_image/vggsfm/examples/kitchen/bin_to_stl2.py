
import os
import numpy as np
import open3d as o3d

def read_point3d_txt(point3d_path):
    if not os.path.exists(point3d_path):
        raise Exception(f"No such file : {point3d_path}")

    with open(point3d_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {point3d_path}")

    comments = lines[:3]
    contents = lines[3:]

    XYZs = []
    RGBs = []
    candidate_ids = {}

    for pt_idx, content in enumerate(contents):
        content_items = content.split(' ')
        pt_id = content_items[0]
        XYZ = content_items[1:4]
        RGB = content_items[4:7]
        error = content_items[7],
        candidate_id = content_items[8::2]
        XYZs.append(np.array(XYZ, dtype=np.float32).reshape(1,3))
        RGBs.append(np.array(RGB, dtype=np.float32).reshape(1, 3) / 255.0)
        candidate_ids[pt_id] = candidate_id
    XYZs = np.concatenate(XYZs, axis=0)
    RGBs = np.concatenate(RGBs, axis=0)

    return XYZs, RGBs, candidate_ids
    
def inverse_relation(candidate_img_ids):
    candidate_point_ids = {}
    pt_ids = list(candidate_img_ids.keys())

    for pt_id in pt_ids:
        candidate_img_id = candidate_img_ids[pt_id]
        for img_id in candidate_img_id:
            if img_id in list(candidate_point_ids.keys()):
                candidate_point_ids[img_id].append(pt_id)
            else:
                candidate_point_ids[img_id] = [pt_id]
    return candidate_point_ids

xyz, colors_rgb, candidateids =read_point3d_txt("points3D.txt")
print(xyz.shape, colors_rgb.shape)
# Create an Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

# Save the PointCloud object to a PCD file
o3d.io.write_point_cloud("points3D.pcd", pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
import open3d as o3d
import trimesh
import numpy as np

pcd.estimate_normals()

# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))

# create the triangular mesh with the vertices and faces from open3d
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                          vertex_normals=np.asarray(mesh.vertex_normals))
o3d.io.write_triangle_mesh("mesh.stl", mesh)
o3d.visualization.draw_geometries([mesh])