import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import trimesh as tm
from tqdm import tqdm
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d
from glob import glob

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num', type=int, default=20)
    parser.add_argument('--exp_path', type=str, default='../data/experiments/exp_test/')
    parser.add_argument('--mesh_path', type=str, default='../data/mesh_example')
    args = parser.parse_args()

    device = 'cpu'

    # load results
    # hand model
    hand_model = HandModel(
        urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
        contact_points_path='allegro_hand_description/contact_points_sgrasp.json', 
        device=device
    )


    grasp_paths = sorted(glob(os.path.join(args.exp_path, 'results') + r'/*.npy'))
    os.makedirs(os.path.join(args.exp_path,f'vis'), exist_ok=True)

    # import pdb;pdb.set_trace()
    for path in grasp_paths:
        isaac_res = np.load(path.replace('.npy','_res.npy').replace('results', 'validated'), allow_pickle=True).item()['sim']

        data_dict_all = np.load(path, allow_pickle=True)
        object_id = path.split('/')[-1].replace('.npy','')
        
        N_vis = range(args.num) if args.num < len(data_dict_all) else range(len(data_dict_all))
        for i in tqdm(N_vis):
            
            data_dict=data_dict_all[i]
            N_object_grasped = len(data_dict['object_code'])
            hand_pose = torch.tensor(data_dict['hand_pose'], dtype=torch.float, device=device)
            object_rotation_matrix = robust_compute_rotation_matrix_from_ortho6d(hand_pose.reshape(N_object_grasped,-1)[:,3:9]).transpose(1, 2)
            object_translation = -torch.bmm(object_rotation_matrix, hand_pose.reshape(N_object_grasped,-1)[:,:3].unsqueeze(2)).squeeze(2)

            # visualize
            standing_pose = torch.cat([torch.tensor([0,0,0]), torch.tensor([1,0,0,0,1,0]), hand_pose[-1][9:]]).to(device)
            hand_model.set_parameters(standing_pose.unsqueeze(0))
            hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='#4A5E65', visual=True)


            object_plotly = []
            for id, object_code in enumerate(data_dict['object_code']):
                object_mesh = tm.load(os.path.join(args.mesh_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False)
                object_scale = data_dict['scale'][id]
                vertices = object_mesh.vertices * object_scale
                

                vertices = vertices @ object_rotation_matrix[id].T.numpy()+object_translation[id].numpy()
                object_plotly.append(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=object_mesh.faces[:, 0], j=object_mesh.faces[:, 1], k=object_mesh.faces[:, 2], 
                                                        opacity=1.0,
                                                        color='pink',
                                                        lighting=dict(ambient=0.8, diffuse=0.6, specular=0., roughness=1., fresnel=0.1)
                                                        ))
            
            fig = go.Figure(hand_en_plotly + object_plotly)
            fig.update_layout(scene_aspectmode='data')
            energy = round(data_dict['energy'][-1],3)
            E_fc = round(data_dict['E_fc'][-1], 3)
            E_dis = round(data_dict['E_dis'][-1], 5)
            E_pen = round(data_dict['E_pen'][-1], 5)
            E_spen = round(data_dict['E_spen'][-1], 5)
            E_joints = round(data_dict['E_joints'][-1], 5)
            result = f'Index {i} energy {energy} E_fc {E_fc}  E_dis {E_dis}  E_pen {E_pen}'
            # fig.add_annotation(text=result, x=0.5, y=-0.1, xref='paper', yref='paper')
            fig.update_layout(
                            scene=dict(
                                camera=dict(
                                    eye=dict(x=2.0, y=0, z=0),  # Camera positioned at (0,1,0)
                                    center=dict(x=0, y=0, z=0)  # Looking at (0,0,0)
                                ),
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                zaxis=dict(visible=False),
                                aspectmode="data"
                            ),
                            title=f'res: {isaac_res[i]}'
                        )
            # fig.show()
            
            os.makedirs(os.path.join(args.exp_path, 'vis', f'imgs_{object_id}',), exist_ok=True)
            fig.write_html(os.path.join(args.exp_path, 'vis', f'imgs_{object_id}', f'{i}.html'), include_plotlyjs='cdn', full_html=False)
            fig.write_image(os.path.join(args.exp_path, 'vis', f'imgs_{object_id}', f'{i}.png'))
