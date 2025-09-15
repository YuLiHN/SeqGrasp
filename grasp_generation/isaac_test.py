from tests.isaac_multiobj_validator import IsaacValidator
import argparse
import torch
import numpy as np
import os
from loguru import logger
from tqdm import tqdm
import random
# from isaac_sim.isaac_hand_model import HandModel
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d, convert_rotation_matrix_to_quat
import trimesh as tm
from glob import glob
from utils.object_model import ObjectModel
from utils.hand_model import HandModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_path', default="../data/mesh_example", type=str)
    parser.add_argument('--eval_dir', default="../data/experiments/exp_test/", type=str)
    parser.add_argument('--gui', action='store_true',default=False)
    parser.add_argument('--pene_thres', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    return args


def read_seqgrasp_result(mesh_path, grasp_path):
    data = np.load(grasp_path, allow_pickle=True)
    obj_names = data[0]['object_code']
    graspem_object_list = ["bulb","camera","cube", "cylinder", "duck", "knob", "mouse", "pear"]
    # import pdb; pdb.set_trace()
    if obj_names[0] in graspem_object_list:
        obj_paths = [os.path.join('../data/objects_graspem', obj_name ) for obj_name in obj_names]
        obj_names = [f"{obj_name}.urdf" for obj_name in obj_names]
    else:
        obj_paths = [os.path.join(mesh_path, obj_name, 'coacd') for obj_name in obj_names]
        obj_names = ["coacd.urdf"] * len(obj_names)

    N_grasp = len(data)
    N_object_grasped = len(data[0]['hand_pose'])
    hand_pose = torch.from_numpy(np.array([it['hand_pose'] for it in data]))

    # N_grasp = q.shape[0]
    quat = torch.tensor([1,0,0,0]).unsqueeze(0).repeat(N_grasp,1) # w,x,y,z
    trans = torch.tensor([0,0,0]).unsqueeze(0).repeat(N_grasp,1)
    joint_values = hand_pose[:,-1,9:].squeeze(1)
    object_scales = [it['scale'] for it in data]
    E_pen = [it['E_pen'][-1] for it in data]
    E_oopen = [it['E_oopen'][-1] for it in data]
    joint_masks = [it['joint_mask'] for it in data]
    oses = [it['os'] for it in data]

    object_rotation_matrix = robust_compute_rotation_matrix_from_ortho6d(hand_pose.reshape(N_grasp*N_object_grasped,-1)[:,3:9]).transpose(1, 2)
    object_translation = -torch.bmm(object_rotation_matrix, hand_pose.reshape(N_grasp*N_object_grasped,-1)[:,:3].unsqueeze(2)).squeeze(2)
    object_quat = convert_rotation_matrix_to_quat(object_rotation_matrix.detach().cpu().numpy()).reshape(N_grasp,N_object_grasped,4)
    object_transl = object_translation.reshape(N_grasp,N_object_grasped,3)
    
    # import pdb; pdb.set_trace()

    return obj_paths, obj_names, N_grasp, quat, trans, joint_values, object_scales, object_transl, object_quat, E_pen, E_oopen, joint_masks, oses



def test_isaac_validator(args):
    # grasp_dict = {}
    if args.gui:
        mode="gui"
    else:
        mode="direct"
    
    # I dont know why we have to use movable but otherwise it will not work
    dexterous_asset_file = "allegro_hand_description_right_movable.urdf"
    assetRoot = "./allegro_hand_description"
    hand_model_name = 'allegro_right'

    
    sim = IsaacValidator(gpu=0, mode=mode, hand_model_name=hand_model_name)
    
    across_all_succ, across_all_cases = 0 ,0
    grasp_paths = sorted(glob(os.path.join(args.eval_dir, 'results') + r'/*'))
    
    for path in tqdm(grasp_paths):

        obj_paths, obj_codes, N_grasp, quat, trans, joint_values, object_scales, object_transl, object_quat, E_pen, E_oopen, joint_masks, oses = read_seqgrasp_result(args.mesh_path, path)
        sim.set_asset(assetRoot, dexterous_asset_file,
                    obj_paths, obj_codes)
        simulated = np.zeros(N_grasp)
        result = []
        for index in range(N_grasp):
            sim.add_env(quat[index], trans[index], joint_values[index],
                            object_scales[index], object_rotation=object_quat[index], object_translation=object_transl[index])
        result = [*result, *sim.run_sim()]
        sim.reset_simulator()

        for i in range(N_grasp):
            simulated[i] = np.array(sum(result[i * 6:(i + 1) * 6]) == 6)
            
        pene = [E_pen[i] + E_oopen[i] for i in range(N_grasp)]
        pene_flag = np.array(pene) < args.pene_thres
        logger.info(f'Success rate of {obj_codes[-1]} : {int(simulated.sum())} / {N_grasp}')
        logger.info(f'Pene filter  : {pene_flag.sum()} / {N_grasp}')
        across_all_succ += simulated.sum()
        across_all_cases += N_grasp
        # import pdb; pdb.set_tra
        save_results(args.eval_dir, path, simulated, pene, args.pene_thres, joint_masks, oses)
    logger.info(f'**Success Rate** across all objects: {across_all_succ} / {across_all_cases}, succ rate: {across_all_succ / across_all_cases}')
    sim.destroy()

def set_global_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_results(eval_dir, grasp_path, result, pene, pene_thres, joint_masks, oses):
    object_code = grasp_path.split('/')[-1].split('.')[0]
    data = np.load(grasp_path, allow_pickle=True)

    finished = []
    for i,jms in enumerate(joint_masks):
        res = np.array(jms).sum(axis=0)[9:]
        # import pdb;pdb.set_trace()
        if res.all(): # no free joints
            finished.append(1)
        else:
            finished.append(0)
    validated_data = []
    finished_data = []
    # import pdb;pdb.set_trace()
    for i, grasp in enumerate(data):
        if result[i] and pene[i]<pene_thres:
            if finished[i]:
                finished_data.append(grasp)
            else:
                validated_data.append(grasp)
    # print("")
    logger.info(f'finished grasps: {len(finished_data)}')
    sim_and_pene = {'sim':result, 'pene':pene}
    print(result)
    os.makedirs(os.path.join(eval_dir, 'validated'), exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'finished'), exist_ok=True)
    np.save(os.path.join(eval_dir, 'validated', object_code + '.npy'), validated_data, allow_pickle=True)
    np.save(os.path.join(eval_dir, 'validated', object_code + '_res' + '.npy'), sim_and_pene, allow_pickle=True)
    np.save(os.path.join(eval_dir, 'finished', object_code + '.npy'), finished_data, allow_pickle=True)

def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    np.random.seed(10)

    logger.add(args.eval_dir + '/evaluation.log')
    logger.info(f'Evaluation directory: {args.eval_dir}')

    logger.info('Start evaluating..')
    test_isaac_validator(args)

    logger.info('End evaluating..')

if __name__ == '__main__':

    main()