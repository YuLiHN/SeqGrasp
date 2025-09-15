"""
Last modified date: 2025.02.11
Author: haofei
Description: initializations
"""

import torch
import transforms3d
import math
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import numpy as np
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from pytorch3d.transforms import matrix_to_axis_angle


    
def initialize_convex_hull(hand_model, object_model, args, init_grasp_params=None):
    device = hand_model.device
    n_objects = len(object_model.object_mesh_list)
    batch_size_each = object_model.batch_size_each
    total_batch_size = n_objects * batch_size_each
    obj_bbx_ratio_list = []

    translation = torch.zeros([total_batch_size, 3], dtype=torch.float, device=device)
    rotation = torch.zeros([total_batch_size, 3, 3], dtype=torch.float, device=device)
    p_list = []
    for i in range(n_objects):
        
        # get inflated convex hull

        mesh_origin = object_model.object_mesh_list[i].convex_hull
        vertices = mesh_origin.vertices.copy()
        faces = mesh_origin.faces
        vertices *= object_model.object_scale_tensor[i].max().item()
        mesh_origin = tm.Trimesh(vertices, faces)
        mesh_origin.faces = mesh_origin.faces[mesh_origin.remove_degenerate_faces()]
        vertices += 0.2 * vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        mesh = tm.Trimesh(vertices=vertices, faces=faces).convex_hull
        vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.float, device=device)
        mesh_pytorch3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

        # sample points
        dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_pytorch3d, num_samples=100 * batch_size_each)
        p = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=batch_size_each)[0][0]
        closest_points, _, _ = mesh_origin.nearest.on_surface(p.detach().cpu().numpy())
        closest_points = torch.tensor(closest_points, dtype=torch.float, device=device)
        n = (closest_points - p) / (closest_points - p).norm(dim=1).unsqueeze(1)


        distance = args.distance_lower + (args.distance_upper - args.distance_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        deviate_theta = args.theta_lower + (args.theta_upper - args.theta_lower) * torch.rand([batch_size_each], dtype=torch.float, device=device)
        process_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)
        rotate_theta = 2 * math.pi * torch.rand([batch_size_each], dtype=torch.float, device=device)

        # solve transformation

        rotation_local = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        rotation_global = torch.zeros([batch_size_each, 3, 3], dtype=torch.float, device=device)
        for j in range(batch_size_each):
            rotation_local[j] = torch.tensor(transforms3d.euler.euler2mat(process_theta[j], deviate_theta[j], rotate_theta[j], axes='rzxz'), dtype=torch.float, device=device)
            rotation_global[j] = torch.tensor(transforms3d.euler.euler2mat(math.atan2(n[j, 1], n[j, 0]) - math.pi / 2, -math.acos(n[j, 2]), 0, axes='rzxz'), dtype=torch.float, device=device)
        translation[i * batch_size_each: (i + 1) * batch_size_each] = p - distance.unsqueeze(1) * (rotation_global @ rotation_local @ torch.tensor([0, 0, 1], dtype=torch.float, device=device).reshape(1, -1, 1)).squeeze(2)
        rotation_hand = torch.tensor(transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz'), dtype=torch.float, device=device)
        rotation[i * batch_size_each: (i + 1) * batch_size_each] = rotation_global @ rotation_local @ rotation_hand
    
    # add something to freeze some of joints
    if init_grasp_params is None:
        joint_angles_mu = torch.tensor( [0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.8, 0.2, 0, 0, ], dtype=torch.float, device=device)
        joint_angles_sigma = args.jitter_strength * (hand_model.joints_upper - hand_model.joints_lower)
        joint_angles = torch.zeros([total_batch_size, hand_model.n_dofs], dtype=torch.float, device=device)
        for i in range(hand_model.n_dofs):
            torch.nn.init.trunc_normal_(joint_angles[:, i], joint_angles_mu[i], joint_angles_sigma[i], hand_model.joints_lower[i] - 1e-6, hand_model.joints_upper[i] + 1e-6)
    else:
        joint_angles = init_grasp_params['prev_hand_pose'][:,-1][:, 6:].to(device)

        

    joint_mask_list = []
    selected_os_list = []
    contact_pool_list = []

    for i in range(total_batch_size):
        
        if args.specific_os != 0:
            potential_selected_os_list = [['link_3.0', 'link_7.0'], ['link_7.0', 'link_11.0'], ['link_3.0', 'link_15.0'],
                        ['base_link', 'link_3.0'], ['base_link', 'link_7.0'], ['base_link', 'link_11.0'], ['base_link', 'link_15.0']]
            
            selected_os = tuple(potential_selected_os_list[args.specific_os-1])
            contact_pool = hand_model.os_contact_pool_dict[selected_os[0]+'+'+selected_os[1]]
                
        else:
            if init_grasp_params is None:
                selected_os, contact_pool, joint_mask_i = hand_model.get_an_os()
            else:
                selected_os, contact_pool, joint_mask_i = hand_model.get_an_os(prev_oses_link=init_grasp_params['prev_os'][i])
        contact_pool_list.append(contact_pool)
        joint_mask_list.append(torch.from_numpy(joint_mask_i).to(device=device, dtype=torch.float))
        selected_os_list.append(selected_os)
    joint_mask = torch.stack(joint_mask_list, dim=0)
    if init_grasp_params is not None :
        # joint_mask = joint_mask * (1.0 - init_grasp_params['prev_joint_mask'][:,9:].to(device)) # freeze the joints that are previously used
        for i in range(init_grasp_params['prev_joint_mask'].shape[1]):
            joint_mask = joint_mask * (1.0 - init_grasp_params['prev_joint_mask'][:,i,9:]).to(device)

    joint_mask = torch.cat([torch.ones([total_batch_size,9],device=device),joint_mask],dim=1)
    hand_model.joint_mask = joint_mask
    hand_model.selected_os = selected_os_list
    hand_model.contact_pool_list = contact_pool_list
    contact_pairs = [[*np.random.choice(sublist[0], args.n_contact//2).tolist(), *np.random.choice(sublist[1], args.n_contact//2).tolist()] for sublist in contact_pool_list]
    contact_point_indices = torch.tensor(contact_pairs).to(device)
    

    hand_pose = torch.cat([
        translation, 
        rotation.transpose(1, 2)[:, :2].reshape(-1, 6),
        joint_angles
    ], dim=1).contiguous()
    hand_pose.requires_grad_()
    hand_model.set_parameters(hand_pose, contact_point_indices)
    if init_grasp_params is not None:
        hand_model.update_grasped_objects(init_grasp_params['prev_object_code_list'], 
                                           init_grasp_params['prev_object_rotation_matrix'], 
                                           init_grasp_params['prev_object_translation'], 
                                           init_grasp_params['prev_object_scale'], 
                                           mesh_path='../data/meshdata', 
                                           num_samples=1024, 
                                           batch_size_each=batch_size_each)
        
    
