"""
Last modified date: 2025.02.11
Author: haofei
Description: energy functions
"""

import torch
import plotly.graph_objects as go



def cal_energy(hand_model, object_model, w_fc=10.,w_dis=100.0, w_pen=50.0, w_spen=10.0, w_joints=1.0,w_oopen=50.0, verbose=False):
    
    # E_dis
    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)

    # E_fc
    contact_normal = contact_normal.reshape(batch_size, 1, 3 * n_contact)
    transformation_matrix = torch.tensor([[0, 0, 0, 0, 0, -1, 0, 1, 0],
                                          [0, 0, 1, 0, 0, 0, -1, 0, 0],
                                          [0, -1, 0, 1, 0, 0, 0, 0, 0]],
                                         dtype=torch.float, device=device)
    g = torch.cat([torch.eye(3, dtype=torch.float, device=device).expand(batch_size, n_contact, 3, 3).reshape(batch_size, 3 * n_contact, 3),
                   (hand_model.contact_points @ transformation_matrix).view(batch_size, 3 * n_contact, 3)], 
                  dim=2).float().to(device)
    norm = torch.norm(contact_normal @ g, dim=[1, 2])
    E_fc = norm * norm
    
    if hand_model.n_dofs == 45: #mano
        E_joints = torch.norm((hand_model.hand_pose[:, 6:] - hand_model.pose_distrib[0]) / hand_model.pose_distrib[1], dim=-1)
    else:
        E_joints = torch.sum((hand_model.hand_pose[:, 9:] > hand_model.joints_upper) * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper), dim=-1) + \
            torch.sum((hand_model.hand_pose[:, 9:] < hand_model.joints_lower) * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]), dim=-1)
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = object_model.surface_points_tensor * object_scale  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)

    E_oopen=torch.zeros(batch_size, dtype=torch.float).to(device)
    if hand_model.grasped_objects_points is not None:
        grasped_objects_points = hand_model.get_grasped_object_points().contiguous() # [n_objects * batch_size_each, n_samples*n_grasped_objects, 3]
        distances_oo,_,closest_points = object_model.cal_distance(grasped_objects_points, with_closest_points=True)
        distances_oo[distances_oo <= 0] = 0 # <0 means outside
        E_oopen = distances_oo.sum(-1)


        

    # E_spen
    E_spen = hand_model.self_penetration()

    if verbose:
        return w_fc * E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_oopen * E_oopen, E_fc, E_dis, E_pen, E_spen, E_joints, E_oopen
    else:
        return w_fc * E_fc + w_dis * E_dis + w_pen * E_pen + w_spen * E_spen + w_joints * E_joints + w_oopen * E_oopen
