"""
Last modified date: 2025.02.11
Author: haofei
Description: Class HandModel
"""

import os

os.chdir(os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import torch
import transforms3d
import trimesh as tm
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
import pytorch_kinematics as pk
from urdf_parser_py.urdf import Robot, Box, Sphere
import pytorch3d.structures
import pytorch3d.ops
import plotly.graph_objects as go
from torchsdf import index_vertices_by_faces, compute_sdf
import random


class HandModel:
    def __init__(self, urdf_path, contact_points_path, n_surface_points=0, device='cpu'):
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float, device=device)
        self.robot = Robot.from_xml_file(urdf_path)
        self.n_dofs = len(self.chain.get_joint_parameter_names())
        
        contact_points = json.load(open(contact_points_path, 'r'))

        self.mesh = {}
        areas = {}
        for link in self.robot.links:
            if link.visual is None or link.collision is None:
                continue
            self.mesh[link.name] = {}
            # load collision mesh
            collision = link.collision
            if type(collision.geometry) == Sphere:
                link_mesh = tm.primitives.Sphere(radius=collision.geometry.radius)
                self.mesh[link.name]['radius'] = collision.geometry.radius
            if type(collision.geometry) == Box:
                # link_mesh = tm.primitives.Box(extents=collision.geometry.size)
                link_mesh = tm.load_mesh(os.path.join(os.path.dirname(urdf_path), 'meshes', 'box.obj'), process=False)
                link_mesh.vertices *= np.array(collision.geometry.size) / 2
            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(collision.geometry, 'scale') and collision.geometry.scale is None:
                collision.geometry.scale = [1, 1, 1]
            scale = torch.tensor(getattr(collision.geometry, 'scale', [1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(getattr(collision.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(collision.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'vertices': vertices,
                'faces': faces,
            })
            if 'radius' not in self.mesh[link.name]:
                self.mesh[link.name]['face_verts'] = index_vertices_by_faces(vertices, faces)
            areas[link.name] = tm.Trimesh(vertices.cpu().numpy(), faces.cpu().numpy()).area.item()
            # load visual mesh
            visual = link.visual
            filename = os.path.join(os.path.dirname(os.path.dirname(urdf_path)), visual.geometry.filename[10:])
            link_mesh = tm.load_mesh(filename)
            vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
            faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
            if hasattr(visual.geometry, 'scale') and visual.geometry.scale is None:
                visual.geometry.scale = [1, 1, 1]
            scale = torch.tensor(getattr(visual.geometry, 'scale', [1, 1, 1]), dtype=torch.float, device=device)
            translation = torch.tensor(getattr(visual.origin, 'xyz', [0, 0, 0]), dtype=torch.float, device=device)
            rotation = torch.tensor(transforms3d.euler.euler2mat(*getattr(visual.origin, 'rpy', [0, 0, 0])),
                                    dtype=torch.float, device=device)
            vertices = vertices * scale
            vertices = vertices @ rotation.T + translation
            self.mesh[link.name].update({
                'visual_vertices': vertices,
                'visual_faces': faces,
            })
            # load contact candidates and penetration keypoints
            contact_candidates = torch.tensor(contact_points[link.name], dtype=torch.float32, device=device).reshape(-1, 3)
            self.mesh[link.name].update({
                'contact_candidates': contact_candidates,
            })

        self.joints_lower = torch.tensor([joint.limit.lower for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        self.joints_upper = torch.tensor([joint.limit.upper for joint in self.robot.joints if joint.joint_type == 'revolute'], dtype=torch.float, device=device)
        
        # sample surface points
        total_area = sum(areas.values())
        num_samples = dict([(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh])
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())
        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]['surface_points'] = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                continue
            mesh = pytorch3d.structures.Meshes(self.mesh[link_name]['vertices'].unsqueeze(0), self.mesh[link_name]['faces'].unsqueeze(0))
            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            self.mesh[link_name]['surface_points'] = surface_points

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))
        self.surface_points_link_indices = torch.cat([self.link_name_to_link_index[link_name] * torch.ones(self.mesh[link_name]['surface_points'].shape[0], dtype=torch.long, device=device) for link_name in self.mesh])
        
        self.contact_candidates = [self.mesh[link_name]['contact_candidates'] for link_name in self.mesh]
        self.global_index_to_link_index = sum([[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)], [])
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        # contact candidates catagorized by link
        self.contact_per_link_index = [torch.sum(self.global_index_to_link_index == i).item() for i in range(len(self.mesh))]
        sum_total=0
        self.contact_lower_index_link_index,self.contact_upper_index_link_index = [],[]
        for i in range(len(self.mesh)):
            self.contact_lower_index_link_index.append(sum_total)
            sum_total += self.contact_per_link_index[i]
            self.contact_upper_index_link_index.append(sum_total)
        self.contact_lower_index_link_index = np.array(self.contact_lower_index_link_index)
        self.contact_upper_index_link_index = np.array(self.contact_upper_index_link_index)
        self.os_contact_pool_dict = self.construct_contact_pool_dict()
        
        # build collision mask
        self.adjacency_mask = torch.zeros([len(self.mesh), len(self.mesh)], dtype=torch.bool, device=device)
        for joint in self.robot.joints:
            parent_id = self.link_name_to_link_index[joint.parent]
            child_id = self.link_name_to_link_index[joint.child]
            self.adjacency_mask[parent_id, child_id] = True
            self.adjacency_mask[child_id, parent_id] = True
        self.adjacency_mask[self.link_name_to_link_index['base_link'], self.link_name_to_link_index['link_13.0']] = True
        self.adjacency_mask[self.link_name_to_link_index['link_13.0'], self.link_name_to_link_index['base_link']] = True

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None
        self.contact_points = None
        self.joint_mask = None
        

        # modified from this line
        self.grasped_objects_points = None
        
        self.os_joint_corres = {
                'base_link+link_3.0': np.array([1.]*4+[0.]*12),
                'base_link+link_7.0': np.array([0.]*4+ [1.]*4+[0.]*8),
                'base_link+link_11.0': np.array([0.]*8+ [1.]*4+[0.]*4),
                'base_link+link_15.0': np.array([0.]*12+ [1.]*4),
                
                'link_3.0+link_7.0': np.array([1.]*8+ [0.]*8),
                'link_7.0+link_11.0': np.array([0.]*4+ [1.]*8+ [0.]*4),
                'link_3.0+link_15.0': np.array([0.]*8+ [1.]*8),
            }

    def set_parameters(self, hand_pose, contact_point_indices=None):
        self.hand_pose = hand_pose
        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()
        self.global_translation = self.hand_pose[:, 0:3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        self.current_status = self.chain.forward_kinematics(self.hand_pose[:, 9:])
        if contact_point_indices is not None:
            self.contact_point_indices = contact_point_indices
            batch_size, n_contact = contact_point_indices.shape
            self.contact_points = self.contact_candidates[self.contact_point_indices]
            link_indices = self.global_index_to_link_index[self.contact_point_indices]
            transforms = torch.zeros(batch_size, n_contact, 4, 4, dtype=torch.float, device=self.device)
            for link_name in self.mesh:
                mask = link_indices == self.link_name_to_link_index[link_name]
                cur = self.current_status[link_name].get_matrix().unsqueeze(1).expand(batch_size, n_contact, 4, 4)
                transforms[mask] = cur[mask]
            self.contact_points = torch.cat([self.contact_points, torch.ones(batch_size, n_contact, 1, dtype=torch.float, device=self.device)], dim=2)
            self.contact_points = (transforms @ self.contact_points.unsqueeze(3))[:, :, :3, 0]
            self.contact_points = self.contact_points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
    
    def cal_distance(self, x):
        # x: (total_batch_size, num_samples, 3)
        # 单独考虑每个link
        # 先把x变换到link的局部坐标系里面，得到x_local: (total_batch_size, num_samples, 3)
        # 然后计算dis，按照内外取符号，内部是正号
        # 最后的dis就是所有link的dis的最大值
        # 对于sphere的link，使用解析方法计算dis，否则用mesh的方法计算dis
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if 'radius' not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]['radius'] - x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis
    
    def cal_num_contact(self, x, threshold=0.0):
        # x: (total_batch_size, num_samples, 3)
        # 单独考虑每个link
        # 先把x变换到link的局部坐标系里面，得到x_local: (total_batch_size, num_samples, 3)
        # 然后计算dis，按照内外取符号，内部是正号
        # 最后的dis就是所有link的dis的最大值
        # 对于sphere的link，使用解析方法计算dis，否则用mesh的方法计算dis
        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)
            if 'radius' not in self.mesh[link_name]:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                dis_local = self.mesh[link_name]['radius'] - x_local.norm(dim=1)
            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        # dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        min_dis = torch.max(torch.stack(dis, dim=0), dim=2)[0]
        # import pdb; pdb.set_trace()
        n_contact = (min_dis >= threshold).sum(dim=0)  
        # import pdb; pdb.set_trace()
        return n_contact
    
    
    def cal_self_distance(self):
        # get surface points
        x = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            x.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != x[-1].shape[0]:
                x[-1] = x[-1].expand(batch_size, n_surface_points, 3)
        x = torch.cat(x, dim=-2).to(self.device)  # (total_batch_size, n_surface_points, 3)
        if len(x.shape) == 2:
            x = x.expand(1, x.shape[0], x.shape[1])
        # cal distance
        dis = []
        for link_name in self.mesh:
            matrix = self.current_status[link_name].get_matrix()
            # if len(matrix) != len(x): 
            #     matrix = matrix.expand(len(x), 4, 4)
            # x_local = transform_points_inverse(x, matrix)
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * n_surface_points, 3)
            if 'radius' in self.mesh[link_name]:
                radius = self.mesh[link_name]['radius']
                dis_local = radius - (x_local.square().sum(-1) + 1e-8).sqrt()  # (total_batch_size * n_surface_points,)
            else:
                face_verts = self.mesh[link_name]['face_verts']
                dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                dis_local = (dis_local + 1e-8).sqrt()
                dis_local = dis_local * (-dis_signs)
            dis_local = dis_local.reshape(x.shape[0], x.shape[1])  # (total_batch_size, n_surface_points)
            is_adjacent = self.adjacency_mask[self.link_name_to_link_index[link_name], self.surface_points_link_indices]  # (n_surface_points,)
            dis_local[:, is_adjacent | (self.link_name_to_link_index[link_name] == self.surface_points_link_indices)] = -float('inf')
            dis.append(dis_local)
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        dis = self.cal_self_distance()
        dis[dis <= 0] = 0
        E_spen = dis.sum(-1)
        return E_spen

    def get_contact_candidates(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['contact_candidates'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['contact_candidates']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    def get_surface_points(self):
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]['surface_points'].shape[0]
            points.append(self.current_status[link_name].transform_points(self.mesh[link_name]['surface_points']))
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points
    
    
    def get_plotly_data(self, i, opacity=0.5, color='lightblue', with_contact_points=False, visual=False):
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]['visual_vertices' if visual else 'vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['visual_faces' if visual else 'faces'].detach().cpu()
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], text=[link_name] * len(v), color=color, opacity=opacity, hovertemplate='%{text}',
                                  lighting=dict(ambient=0.8, diffuse=0.5, specular=0., roughness=1.,)
                                  ))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2],
                                     mode='markers', marker=dict(color='red', size=5)))
        return data
    
    def get_plotly_data_real(self, i, opacity=0.5, with_contact_points=False, visual=True):
        data = []
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]['visual_vertices' if visual else 'vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['visual_faces' if visual else 'faces'].detach().cpu()
            if 'tip' in link_name:
                color='white'
            else:
                # color='#2B2B2B'
                color='#222222' #black but with edges
            data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], 
                                  text=[link_name] * len(v), color=color, opacity=opacity, hovertemplate='%{text}',
                                  lighting=dict(
                                                    ambient=0.1,
                                                    diffuse=1.0,
                                                    specular=0.2,
                                                    roughness=0.7,
                                                    fresnel=0.3
                                                ),
                                            lightposition=dict(
                                                x=10,  # Light source position
                                                y=10,
                                                z=15)))
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            data.append(go.Scatter3d(x=contact_points[:, 0], y=contact_points[:, 1], z=contact_points[:, 2],
                                     mode='markers', marker=dict(color='red', size=5)))
        return data
    def get_trimesh_data(self, i):
        """
        Get full mesh
        
        Returns
        -------
        data: trimesh.Trimesh
        """
        data = tm.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(
                self.mesh[link_name]['vertices'])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]['faces'].detach().cpu()
            data += tm.Trimesh(vertices=v, faces=f)
        return data

    # add functions from this line:
    def update_grasped_objects(self, object_codes, object_rotation, object_translation, object_scale, mesh_path, num_samples, batch_size_each):
        # this function assume that every grasp has the same previous objects
        graspem_object_list = ["bulb","camera","cube", "cylinder", "duck", "knob", "mouse", "pear"]
        self.N_grasped_objects = len(object_codes[0])
        self.N_objects = len(object_codes)
        
        object_points_batch = []
        for i_object in object_codes:
            object_points = []
            for object_code in i_object:
                # mesh = tm.load(os.path.join(mesh_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False)
                if object_code in graspem_object_list:
                    mesh = tm.load(os.path.join(mesh_path, object_code, f'{object_code}.stl'))
                else:
                    mesh = tm.load(os.path.join(mesh_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False)
                vertices = torch.tensor(mesh.vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(mesh.faces, dtype=torch.float, device=self.device)
                mesh_py3d = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))
                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh_py3d, num_samples=100 * num_samples)
                surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples)[0][0]
                surface_points.to(dtype=float, device=self.device)
                object_points.append(surface_points)
                
            grasped_objects_points = torch.stack(object_points).to(self.device).unsqueeze(0).repeat(batch_size_each, 1, 1, 1) # [batch_size_each, N_grasped_objects, N_points, 3]
            object_points_batch.append(grasped_objects_points)
        self.grasped_objects_points = torch.cat(object_points_batch, dim=0)
        self.grasped_objects_points = self.grasped_objects_points * torch.tensor(object_scale).to(self.device).unsqueeze(-1).unsqueeze(-1)# scale it
        self.grasped_objects_points = torch.einsum('b o i j, b o p j -> b o p i',  object_rotation.to(self.device), self.grasped_objects_points.to(self.device)) + object_translation.unsqueeze(2).to(self.device) #transform the points
        # stack all grasped object pc cause they are rigid forever
        self.grasped_objects_points = self.grasped_objects_points.reshape(batch_size_each*self.N_objects, num_samples*self.N_grasped_objects,3)
    
    def get_grasped_object_points(self):
        
        # import pdb;pdb.set_trace()
        if self.grasped_objects_points is None: raise Exception("Sorry, use update_grasped_objects() first!")
        # N_objects = self.global_rotation.shape[0] // self.grasped_objects_points.shape[0]
        points = torch.einsum('b i j, b p j -> b p i', 
                              self.global_rotation, 
                              self.grasped_objects_points) 
        return points + self.global_translation.unsqueeze(1)
    # def find_all
    def find_all_parent_joints(self, link_name):
        parent_joints = []
        for joint in self.robot.joints:
            if joint.child == link_name:
                parent_joints.append(joint.parent)
        return parent_joints
    
    def construct_contact_pool_dict(self, hand_model='allegro_right'):
        
        if hand_model == 'allegro_right':
            contact_group_dict = {}
            triple_side_links = ['base_link', 'link_3.0','link_2.0','link_1.0', 'link_7.0', 'link_6.0', 'link_5.0']
            double_side_links = ['link_11.0','link_10.0','link_9.0', 'link_15.0','link_14.0']
            for link_name in self.mesh:
                contact_group_dict[link_name] = {}
                lower = self.contact_lower_index_link_index[self.link_name_to_link_index[link_name]]
                upper = self.contact_upper_index_link_index[self.link_name_to_link_index[link_name]]
                if link_name in triple_side_links:
                    contact_group_dict[link_name]['side1'] = list(range(lower, lower+(upper-lower)//3))
                    contact_group_dict[link_name]['side2'] = list(range(lower+(upper-lower)//3, lower+(upper-lower)//3*2))
                    contact_group_dict[link_name]['side3'] = list(range(lower+(upper-lower)//3*2, upper))
                if link_name in double_side_links:
                    contact_group_dict[link_name]['side1'] = list(range(lower, lower+(upper-lower)//2))
                    contact_group_dict[link_name]['side2'] = list(range(lower+(upper-lower)//2, upper))
            # import pdb;pdb.set_trace()
            os_contact_pool_dict = {
                                        'base_link+link_3.0': (contact_group_dict['base_link']['side1'] , contact_group_dict['link_3.0']['side2']),
                                        'base_link+link_7.0': (contact_group_dict['base_link']['side2'] , contact_group_dict['link_7.0']['side2']),
                                        'base_link+link_11.0': (contact_group_dict['base_link']['side3'] , contact_group_dict['link_11.0']['side1']),
                                        'base_link+link_15.0': (contact_group_dict['base_link']['side1'] , contact_group_dict['link_15.0']['side1']),
                                        
                                        
                                        'link_3.0+link_7.0': (contact_group_dict['link_3.0']['side1']+contact_group_dict['link_2.0']['side1']+contact_group_dict['link_1.0']['side1'], 
                                                            contact_group_dict['link_7.0']['side3']+contact_group_dict['link_6.0']['side3']+contact_group_dict['link_5.0']['side3']),
                                        
                                        'link_7.0+link_11.0': (contact_group_dict['link_7.0']['side1']+contact_group_dict['link_6.0']['side1']+contact_group_dict['link_5.0']['side1'], 
                                                            contact_group_dict['link_11.0']['side2']+contact_group_dict['link_10.0']['side2']+contact_group_dict['link_9.0']['side2']),
                                        
                                        # 'link_3.0+link_15.0': (contact_group_dict['link_3.0']['side1']+contact_group_dict['link_3.0']['side2']+contact_group_dict['link_3.0']['side3'], 
                                        #                     contact_group_dict['link_15.0']['side2']+contact_group_dict['link_15.0']['side1']),
                                        'link_3.0+link_15.0': (contact_group_dict['link_3.0']['side2'], 
                                                            contact_group_dict['link_15.0']['side1']),
                                        'link_3.0+link_15.0_side': (contact_group_dict['link_3.0']['side3']+ contact_group_dict['link_2.0']['side3'], 
                                                            contact_group_dict['link_15.0']['side1']),
                                    }
        else:
            raise NotImplementedError
        return os_contact_pool_dict
    
    def get_an_os(self, object_bbx_ratio=0., prev_oses_link=None, hand_name='allegro_right'):
        # get a good os for the object
        # if prev_os is not None, return the next os
        # else return a random os
        # finger_links = ['base_link', 'link_3.0', 'link_7.0', 'link_11.0', 'link_15.0']
        if hand_name=='allegro_right':
            ratio_thres=1.0
            palm_finger_pool = [('base_link+link_3.0'), ('base_link+link_7.0'), ('base_link+link_11.0'), ('base_link+link_15.0')]
            finger_finger_pool = [('link_3.0+link_7.0'), ('link_7.0+link_11.0'), ('link_3.0+link_15.0')]
            
            next_palm_finger_pool = palm_finger_pool.copy()
            next_finger_finger_pool = finger_finger_pool.copy()
            if prev_oses_link is not None:
                joint_mask_his = [self.os_joint_corres[os_i] for os_i in prev_oses_link]
                joint_mask_his = (1.-np.array(joint_mask_his)).sum(axis=0)
                for os in next_palm_finger_pool:
                    joint_mask_curr = self.os_joint_corres[os]*joint_mask_his
                    if joint_mask_curr.sum()==0.:
                        next_palm_finger_pool.remove(os)
                for os in next_finger_finger_pool:
                    joint_mask_curr = self.os_joint_corres[os]*joint_mask_his
                    if joint_mask_curr.sum()==0:
                        next_finger_finger_pool.remove(os)
            else:
                joint_mask_his = np.ones(16)
                        
            if object_bbx_ratio>ratio_thres and next_finger_finger_pool:
                selected_os = random.choice(next_finger_finger_pool)
            else:
                selected_os = random.choice(next_palm_finger_pool+next_finger_finger_pool)

            contact_pool = self.os_contact_pool_dict[selected_os]
            return selected_os, contact_pool, self.os_joint_corres[selected_os]*joint_mask_his

        else:
            raise NotImplementedError
                
        