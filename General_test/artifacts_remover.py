import pymeshlab
import open3d as o3d
import numpy as np
import os

def remove_black_faces(mesh_path, output_path, black_threshold=0.2): #maybe create a threshold in dependency from the overall color 
                                                                       #of the input images!!
                                                                       # 0.025 works well for objects with black in the image
                                                                       #0.1 works well for objects with no black in the image
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_triangle_normals()
    
    if not mesh.has_vertex_colors():
        print("Mesh does not have vertex colors.")
        return

    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    non_black_vertices_mask = np.any(colors >= black_threshold, axis=1)
    
    # Filter vertices and get new indices
    new_vertex_indices = np.cumsum(non_black_vertices_mask) - 1
    valid_vertices = vertices[non_black_vertices_mask]
    valid_colors = colors[non_black_vertices_mask]

    # Update triangle indices and filter out triangles with any black vertices
    valid_triangles = []
    for triangle in triangles:
        if all(non_black_vertices_mask[vertex_idx] for vertex_idx in triangle):
            new_triangle = [new_vertex_indices[vertex_idx] for vertex_idx in triangle]
            valid_triangles.append(new_triangle)

    # Create a new mesh with filtered vertices and reindexed triangles
    filtered_mesh = o3d.geometry.TriangleMesh()
    filtered_mesh.vertices = o3d.utility.Vector3dVector(valid_vertices)
    filtered_mesh.vertex_colors = o3d.utility.Vector3dVector(valid_colors)
    filtered_mesh.triangles = o3d.utility.Vector3iVector(np.array(valid_triangles, dtype=np.int32))

    # Save the filtered mesh
    o3d.io.write_triangle_mesh(output_path, filtered_mesh)
    #print(f"Filtered mesh saved to: {output_path}")

# Create a MeshSet object
ms = pymeshlab.MeshSet()
mesh_path = '/workspace/data/data_reconstruction/cat_benchmarks/mid_view/4x_90deg/Outputs/sugar/sugarmesh_3Dgs7000_sdfestim02_sdfnorm02_level03_decim1000000.ply'
save_path = os.path.dirname(mesh_path)


# Load the mesh (make sure the mesh has color data)
ms.load_new_mesh(mesh_path)

# Apply the 'Invert Faces Orientation' filter
ms.apply_filter('meshing_invert_face_orientation', forceflip=True, onlyselected=False)

# Check if face color data exists, otherwise transfer vertex colors to faces
try:
    # Try accessing the face color matrix
    face_color_matrix = ms.current_mesh().face_color_matrix()
    print("Face color matrix found.")
except pymeshlab.pmeshlab.MissingComponentException:
    print("Face colors missing, transferring vertex colors to faces.")
    ms.apply_filter('compute_color_transfer_vertex_to_face')
    ms.apply_filter('compute_color_from_texture_per_vertex') 

# Check dimensions of the face color matrix
print(np.shape(ms.current_mesh().face_color_matrix()))
# Save the modified mesh to a new file
temp_path = save_path + '/temporary.ply'
ms.save_current_mesh(temp_path)
# Apply the selection filter based on black faces (RGB values close to [0,0,0])
remove_black_faces(temp_path, temp_path)


# Create a MeshSet object
ms = pymeshlab.MeshSet()
# Load the mesh (make sure the mesh has color data) 
ms.load_new_mesh(temp_path)
#this works but should be done at the end
ms.apply_filter('meshing_remove_connected_component_by_face_number', mincomponentsize= 200) 
ms.save_current_mesh(save_path + '/artifacts_removed.ply')
