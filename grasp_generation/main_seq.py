"""
Last modified date: 2025.02.11
Author: haofei
Description: Entry of the program
"""

import os

# os.chdir(os.path.dirname(__file__))

import argparse
import shutil
import numpy as np
import torch
from tqdm import tqdm
import math
import transforms3d


from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.logger import Logger
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

import random
from itertools import cycle


def extend(input, M):
    if isinstance(input, list):
        return [x for x, _ in zip(cycle(input), range(M))]
    elif isinstance(input, torch.Tensor):
        return torch.cat([input for _ in range(M//input.shape[0]+1)], dim=0)[:M]



# prepare arguments
def parse_args():
    parser = argparse.ArgumentParser()
    # experiment settings
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', default="0", type=str)
    parser.add_argument('--hand', default="allegro_right", type=str)
    parser.add_argument('--scale', default=[0.04], nargs='+', type=float)
    parser.add_argument('--specific_os', default=0, type=int)
    parser.add_argument('--experiments_dir', default='experiments', type=str)
    parser.add_argument('--object_code_list', nargs='+', default=
        [
        # 'ddg-ycb_011_banana',
        'ddg-ycb_013_apple'
        ])
    parser.add_argument('--prev_grasp_path_list',nargs='+', default= 
                        None, 
                        )
    parser.add_argument('--name', default='exp_test', type=str)
    parser.add_argument('--n_contact', default=2, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--n_iter', default=6000, type=int)
    # hyper parameters
    parser.add_argument('--switch_possibility', default=0.3, type=float)
    parser.add_argument('--mu', default=0.98, type=float)
    parser.add_argument('--eps', default=1e-6, type=float)
    parser.add_argument('--noise_size', default=0.005, type=float)
    parser.add_argument('--stepsize_period', default=50, type=int)
    parser.add_argument('--starting_temperature', default=32, type=float)
    parser.add_argument('--annealing_period', default=30, type=int)
    parser.add_argument('--temperature_decay', default=0.95, type=float)
    parser.add_argument('--w_fc', default=50.0, type=float)
    parser.add_argument('--w_dis', default=50.0, type=float)
    parser.add_argument('--w_pen', default=5.0, type=float)
    parser.add_argument('--w_spen', default=5.0, type=float)
    parser.add_argument('--w_joints', default=1.0, type=float)
    parser.add_argument('--w_oopen', default=5.0, type=float)
    
    # initialization settings
    parser.add_argument('--jitter_strength', default=0.05, type=float)
    parser.add_argument('--distance_lower', default=0.2, type=float)
    parser.add_argument('--distance_upper', default=0.3, type=float)
    parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
    parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
    # energy thresholds
    parser.add_argument('--thres_fc', default=0.3, type=float)
    parser.add_argument('--thres_dis', default=0.005, type=float)
    parser.add_argument('--thres_pen', default=0.001, type=float)

    args = parser.parse_args()
    return args


def main(args):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    np.seterr(all='raise')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    # prepare models

    total_batch_size = len(args.object_code_list) * args.batch_size

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on', device)

    if args.hand == 'allegro_right':
        from utils.hand_model import HandModel
        hand_model = HandModel(
            urdf_path='allegro_hand_description/allegro_hand_description_right.urdf',
            contact_points_path='allegro_hand_description/contact_points_sgrasp.json', 
            n_surface_points=1000, 
            device=device
        )

    else:
        raise ValueError('Unknown hand model!')

    object_model = ObjectModel(
        data_root_path='../data/mesh_example',
        batch_size_each=args.batch_size,
        num_samples=2000, 
        device=device,
        scale=args.scale
    )
    object_model.initialize(args.object_code_list)

    # initialize previous grasp
    prev_grasp_params = {"object_code":[],
                         "hand_pose":[],
                         "os":[],
                         "scale":[],
                         "energy":[],
                         "E_fc":[],
                         "E_dis":[],
                         "E_pen":[],
                         "E_spen":[],
                         "E_joints":[],
                        #  "E_normal":[],
                         "E_oopen":[]}
    if args.prev_grasp_path_list is not None:
        
        assert len(args.prev_grasp_path_list) == len(args.object_code_list)
        prev_joint_mask, prev_hand_pose, prev_object_rotation_matrix, prev_object_translation = [], [], [], []
        
        for prev_grasp_path in args.prev_grasp_path_list:
            prev_grasp_id = np.load(prev_grasp_path, allow_pickle=True)
            N_prev_grasp = len(prev_grasp_id)
            N_grasped_object = len(prev_grasp_id[0]['object_code'])
            N_hand_params = prev_grasp_id[0]['hand_pose'][0].shape[0]

            prev_joint_mask_id = torch.from_numpy(np.array([it['joint_mask'] for it in prev_grasp_id]))
            prev_hand_pose_id = torch.from_numpy(np.array([it['hand_pose'] for it in prev_grasp_id])).reshape(N_prev_grasp*N_grasped_object,N_hand_params)
            prev_object_rotation_matrix_id = robust_compute_rotation_matrix_from_ortho6d(prev_hand_pose_id[:,3:9]).transpose(1, 2)
            prev_object_translation_id = -torch.bmm(prev_object_rotation_matrix_id, prev_hand_pose_id[:,:3].unsqueeze(2)).squeeze(2)

            prev_object_rotation_matrix_id = extend(prev_object_rotation_matrix_id.reshape(N_prev_grasp,N_grasped_object, 3, 3), args.batch_size)
            prev_object_translation_id = extend(prev_object_translation_id.reshape(N_prev_grasp,N_grasped_object, 3), args.batch_size)

            prev_joint_mask.append(extend(prev_joint_mask_id.reshape(N_prev_grasp, N_grasped_object, N_hand_params), args.batch_size))
            prev_hand_pose.append(extend(prev_hand_pose_id.reshape(N_prev_grasp, N_grasped_object, N_hand_params), args.batch_size))
            prev_object_rotation_matrix.append(prev_object_rotation_matrix_id)
            prev_object_translation.append(prev_object_translation_id)

            for key in prev_grasp_params.keys():
                if key != 'object_code':
                    prev_grasp_params[key] += extend([it[key] for it in prev_grasp_id], args.batch_size)
                else:
                    prev_grasp_params[key] += [prev_grasp_id[0][key]] # they are same
        


        init_grasp_params = dict(
            prev_joint_mask=torch.cat(prev_joint_mask,dim=0),
            prev_hand_pose=torch.cat(prev_hand_pose,dim=0),
            prev_os=prev_grasp_params['os'],
            prev_object_code_list=prev_grasp_params['object_code'],
            prev_object_rotation_matrix=torch.cat(prev_object_rotation_matrix,dim=0),
            prev_object_translation=torch.cat(prev_object_translation,dim=0),
            prev_object_scale = prev_grasp_params['scale'],)
    else:
        init_grasp_params = None
        
    initialize_convex_hull(hand_model, object_model, args, init_grasp_params)

    print('n_contact_candidates', hand_model.n_contact_candidates)
    print('total batch size', total_batch_size)
    # hand_pose_st = hand_model.hand_pose.detach()

    optim_config = {
        'switch_possibility': args.switch_possibility,
        'starting_temperature': args.starting_temperature,
        'temperature_decay': args.temperature_decay,
        'annealing_period': args.annealing_period,
        'noise_size': args.noise_size,
        'stepsize_period': args.stepsize_period,
        'mu': args.mu,
        'device': device,
    }
    optimizer = Annealing(hand_model, **optim_config)

    try:
        shutil.rmtree(os.path.join(f'../data/{args.experiments_dir}', args.name, 'logs'))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(f'../data/{args.experiments_dir}', args.name, 'logs'), exist_ok=True)
    logger_config = {
        'thres_fc': args.thres_fc,
        'thres_dis': args.thres_dis,
        'thres_pen': args.thres_pen
    }
    logger = Logger(log_dir=os.path.join(f'../data/{args.experiments_dir}', args.name, 'logs'), **logger_config)


    # optimize

    weight_dict = dict(
        w_fc=args.w_fc,
        w_dis=args.w_dis,
        w_pen=args.w_pen,
        w_spen=args.w_spen,
        w_joints=args.w_joints,
        w_oopen=args.w_oopen,
    )
    energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_oopen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

    energy.sum().backward(retain_graph=True)
    logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_oopen, 0, show=True)
    
    for step in tqdm(range(1, args.n_iter + 1), desc='optimizing'):
        # only use part of the joints
        s = optimizer.try_step(joint_mask=hand_model.joint_mask)

        optimizer.zero_grad()
        weight_dict['w_pen'] = args.w_pen*np.exp(2*step/args.n_iter).item()
        new_energy, new_E_fc, new_E_dis, new_E_pen, new_E_spen, new_E_joints, new_E_oopen = cal_energy(hand_model, object_model, verbose=True, **weight_dict)

        new_energy.sum().backward(retain_graph=True)

        with torch.no_grad():
            accept, t = optimizer.accept_step(energy, new_energy)

            energy[accept] = new_energy[accept]
            E_dis[accept] = new_E_dis[accept]
            E_fc[accept] = new_E_fc[accept]
            E_pen[accept] = new_E_pen[accept]
            E_spen[accept] = new_E_spen[accept]
            E_joints[accept] = new_E_joints[accept]
            # E_normal[accept] = new_E_normal[accept]
            E_oopen[accept] = new_E_oopen[accept]

            logger.log(energy, E_fc, E_dis, E_pen, E_spen, E_joints, E_oopen, step, show=False)


    # save results

    joint_names = [
        'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0',      # index
        'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0',      # middle
        'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',    # ring
        'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'   # thumb
    ]
    try:
        shutil.rmtree(os.path.join(f'../data/{args.experiments_dir}', args.name, 'results'))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(f'../data/{args.experiments_dir}', args.name, 'results'), exist_ok=True)
    result_path = os.path.join(f'../data/{args.experiments_dir}', args.name, 'results')
    # os.makedirs(result_path, exist_ok=True)
    for i in range(len(args.object_code_list)):
        data_list = []
        for j in range(args.batch_size):
            idx = i * args.batch_size + j
            scale = object_model.object_scale_tensor[i][j].item()
            hand_pose = hand_model.hand_pose[idx].detach().cpu()
            
            if args.prev_grasp_path_list is not None:
                prev_scale_i = prev_grasp_params['scale'][idx]
                prev_os_i = prev_grasp_params['os'][idx]
                prev_object_code_i = prev_grasp_params['object_code'][i] # only you
                prev_hand_pose_i = prev_grasp_params['hand_pose'][idx]
                # prev_hand_pose_st_id = prev_hand_pose_st[idx]
                
                prev_energy_i = prev_grasp_params['energy'][idx]
                prev_E_fc_i = prev_grasp_params['E_fc'][idx]
                prev_E_dis_i = prev_grasp_params['E_dis'][idx]
                prev_E_pen_i = prev_grasp_params['E_pen'][idx]
                prev_E_spen_i = prev_grasp_params['E_spen'][idx]
                prev_E_joints_i = prev_grasp_params['E_joints'][idx]
                # prev_E_normal_i = prev_grasp_params['E_normal'][idx]
                prev_E_oopen_i = prev_grasp_params['E_oopen'][idx]
                prev_joint_mask_i = [prev_jm for prev_jm in init_grasp_params['prev_joint_mask'][idx,:].detach().cpu().numpy()]
                
                    
            else:
                prev_scale_i = []
                prev_os_i = []
                prev_object_code_i = []
                prev_hand_pose_i = []
                
                prev_energy_i = []
                prev_E_fc_i = []
                prev_E_dis_i = []
                prev_E_pen_i = []
                prev_E_spen_i = []
                prev_E_joints_i = []
                # prev_E_normal_i = []
                prev_E_oopen_i = []
                prev_joint_mask_i = []
                    
                    
            data_list.append(dict(
                scale=prev_scale_i+[scale],
                # qpos=qpos,
                hand_pose=prev_hand_pose_i+[hand_pose.numpy()],
                # hand_pose_st=prev_hand_pose_st_id+[hand_pose_st_id.numpy()],
                # qpos_st=qpos_st,
                energy=prev_energy_i+[energy[idx].item()],
                E_fc=prev_E_fc_i+[E_fc[idx].item()],
                E_dis=prev_E_dis_i+ [E_dis[idx].item()],
                E_pen=prev_E_pen_i+[E_pen[idx].item()],
                E_spen=prev_E_spen_i+[E_spen[idx].item()],
                E_joints=prev_E_joints_i+[E_joints[idx].item()],
                # E_normal=prev_E_normal_i+[E_normal[idx].item()],
                E_oopen=prev_E_oopen_i+[E_oopen[idx].item()], 
                joint_mask=prev_joint_mask_i+[hand_model.joint_mask[idx].cpu().numpy()],
                os=prev_os_i+[hand_model.selected_os[idx]],
                object_code=prev_object_code_i+[args.object_code_list[i]],
            ))
            
        np.save(os.path.join(result_path, f'{i}_'+args.object_code_list[i] + '.npy'), data_list, allow_pickle=True)

if __name__ == '__main__':
    args = parse_args()
    main(args)