"""
Last modified date: 2025.02.11
Author: haofei
Description: visualize hand model
"""

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.realpath('.'))

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from utils.hand_model import HandModel
from utils.plotly_utils import plot_mesh


torch.manual_seed(1)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    device = torch.device('cpu')

    # hand model

    hand_model = HandModel(
        urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
        contact_points_path='allegro_hand_description/contact_points_sgrasp.json', 
        n_surface_points=1000, 
        device=device
    )
    rot = transforms3d.euler.euler2mat(-np.pi / 2, -np.pi / 2, 0, axes='rzyz')
    hand_pose = torch.cat([
        torch.tensor([0, 0, 0], dtype=torch.float, device=device), 
        torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device=device),
        torch.tensor([0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0.8, 0.2, 0, 0, ], dtype=torch.float, device=device), 
    ], dim=0)
    hand_model.set_parameters(hand_pose.unsqueeze(0))

    contact_candidates = hand_model.get_contact_candidates()[0]
    print(f'n_dofs: {hand_model.n_dofs}')
    print(f'n_contact_candidates: {len(contact_candidates)}')
    print(hand_model.chain.get_joint_parameter_names())
    print("upper bound: ", hand_model.joints_upper)
    print("lower bound: ", hand_model.joints_lower)

    # visualize
    link1_index = [hand_model.link_name_to_link_index[i] for i in ['link_3.0']]
    link1_lower = hand_model.contact_lower_index_link_index[link1_index]
    link1_upper = hand_model.contact_upper_index_link_index[link1_index]
    selected_os_list = [['link_3.0', 'link_7.0'], ['link_7.0', 'link_11.0'], ['link_3.0', 'link_15.0'],
                        ['base_link', 'link_3.0'], ['base_link', 'link_7.0'], ['base_link', 'link_11.0'], ['base_link', 'link_15.0']]

    v_1 = []
    v_2 = []    
    for selected_os in selected_os_list[:3]:
        contact_pool = hand_model.os_contact_pool_dict[selected_os[0]+'+'+selected_os[1]]
        v_1.append(contact_candidates[contact_pool[0]])
        v_1.append(contact_candidates[contact_pool[1]])
        
    for selected_os in selected_os_list[3:7]:
        contact_pool = hand_model.os_contact_pool_dict[selected_os[0]+'+'+selected_os[1]]
        v_2.append(contact_candidates[contact_pool[0]])
        v_2.append(contact_candidates[contact_pool[1]])
    v = contact_candidates.detach().cpu()
        
    hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='grey')
    

    v_1 = torch.cat(v_1).detach().cpu()
    v_2 = torch.cat(v_2).detach().cpu()
    
    contact_candidates_plotly1 = [go.Scatter3d(x=v_1[:, 0], y=v_1[:, 1], z=v_1[:, 2], mode='markers', marker=dict(size=2, color='pink'))]
    contact_candidates_plotly2 = [go.Scatter3d(x=v_2[:, 0], y=v_2[:, 1], z=v_2[:, 2], mode='markers', marker=dict(size=2, color='lightblue'))]

    fig = go.Figure(hand_plotly + 
                    contact_candidates_plotly1+
                    contact_candidates_plotly2)

    fig.update_layout(
                        scene=dict(
                            camera=dict(
                                eye=dict(x=2.0, y=-1.0, z=1.0),  # Camera positioned at (0,1,0)
                                center=dict(x=0, y=0, z=0)  # Looking at (0,0,0)
                            ),
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False),
                            zaxis=dict(visible=False),
                            aspectmode="data"
                        )
                        )
    
    fig.write_image(f'./allegro_contacts.png')
    fig.show()
