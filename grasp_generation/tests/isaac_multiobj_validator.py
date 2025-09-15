"""
Last modified date: 2025.02.11
Author: Haofei
Description: Class IsaacValidator
"""

from isaacgym import gymapi
from isaacgym import gymutil
import math
from time import sleep

gym = gymapi.acquire_gym()


class IsaacValidator():

    def __init__(self,
                 mode='direct',
                 hand_friction=2.0,
                 obj_friction=2.0,
                 threshold_dis=0.1,
                 env_batch=1,
                 sim_step=100,
                 gpu=0,
                 debug_interval=0.05,
                 hand_model_name='allegro_right'):

        self.hand_friction = hand_friction
        self.obj_friction = obj_friction
        self.debug_interval = debug_interval
        self.threshold_dis = threshold_dis
        self.env_batch = env_batch
        self.gpu = gpu
        self.sim_step = sim_step
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        
        if hand_model_name == 'allegro_right':
            self.joint_names = ['joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
                                'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
                                'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0',
                                'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0', 
            ]
        elif hand_model_name == 'shadowhand':
            self.joint_names = ['FFJ4', 'FFJ3', 'FFJ2', 'FFJ1', 
                                'MFJ4', 'MFJ3', 'MFJ2', 'MFJ1', 
                                'RFJ4', 'RFJ3', 'RFJ2', 'RFJ1', 
                                'LFJ5', 'LFJ4', 'LFJ3', 'LFJ2', 
                                'LFJ1', 'THJ5', 'THJ4', 'THJ3', 
                                'THJ2', 'THJ1']
            
        else:
            raise ValueError(f"Hand model {hand_model_name} not supported")
        
        self.hand_asset = None
        self.obj_asset = None
        self.obj_asset_list = []
        self.obj_init_poses = []
        
        self.sim_params = gymapi.SimParams()

        # set common parameters
        self.sim_params.dt = 1 / 60
        self.sim_params.substeps = 2
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0)

        # set PhysX-specific parameters
        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 0
        self.sim_params.physx.contact_offset = 0.002
        self.sim_params.physx.rest_offset = 0.0
        # self.sim_params.physx.default_buffer_size_multiplier = 1024
        # self.sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP


        self.sim_params.use_gpu_pipeline = False
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX,
                                  self.sim_params)
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 800
        self.camera_props.height = 600
        self.camera_props.use_collision_geometry = True

        # set viewer
        self.viewer = None
        if mode == "gui":
            self.has_viewer = True
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
            gym.viewer_camera_look_at(self.viewer, None, gymapi.Vec3(0, 0, 1),
                                      gymapi.Vec3(0, 0, 0))
        else:
            self.has_viewer = False

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.density = 10000
        self.obj_asset_options = gymapi.AssetOptions()
        self.obj_asset_options.override_com = True
        self.obj_asset_options.override_inertia = True
        self.obj_asset_options.density = 500

        self.test_rotations = [
            gymapi.Transform(gymapi.Vec3(0, 0, 0), gymapi.Quat(0, 0, 0, 1)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            1 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1),
                                            -0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            0.5 * math.pi)),
            gymapi.Transform(
                gymapi.Vec3(0, 0, 0),
                gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0),
                                            -0.5 * math.pi)),
        ]
        

    def set_asset(self, hand_root, hand_file, obj_root, obj_file):
        self.hand_asset = gym.load_asset(self.sim, hand_root, hand_file,
                                         self.hand_asset_options)
        for i in range(len(obj_root)):
            self.obj_asset_list.append(gym.load_asset(self.sim, obj_root[i], obj_file[i],
                                            self.obj_asset_options))

    def add_env(self, hand_rotation, hand_translation, hand_qpos, obj_scales, object_rotation,object_translation, target_qpos=None, rel_path=False):
        for test_rot in self.test_rotations:
            env = gym.create_env(self.sim, gymapi.Vec3(-1, -1, -1),
                                 gymapi.Vec3(1, 1, 1), 6)
            self.envs.append(env)
            pose = gymapi.Transform()
            pose.r = gymapi.Quat(*hand_rotation[1:], hand_rotation[0])
            pose.p = gymapi.Vec3(*hand_translation)
            pose = test_rot * pose
            hand_actor_handle = gym.create_actor(
                env, self.hand_asset, pose, "allegro", 0, -1)
            self.hand_handles.append(hand_actor_handle)
            hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
            hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
            hand_props["stiffness"].fill(1000)
            hand_props["damping"].fill(0.0)
            gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
            dof_states = gym.get_actor_dof_states(env, hand_actor_handle,
                                                  gymapi.STATE_ALL)
            for i, joint in enumerate(self.joint_names):
                joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,
                                                     joint,
                                                     gymapi.DOMAIN_ACTOR)
                # if rel_path:
                #     dof_states["pos"][joint_idx] = hand_qpos[i]-0.3
                # else:
                # import pdb; pdb.set_trace()
                dof_states["pos"][joint_idx] = hand_qpos[i]
            gym.set_actor_dof_states(env, hand_actor_handle, dof_states,
                                     gymapi.STATE_ALL)
            if target_qpos != None:
                for i, joint in enumerate(self.joint_names):
                    joint_idx = gym.find_actor_dof_index(env, hand_actor_handle,
                                                         joint,
                                                         gymapi.DOMAIN_ACTOR)
                    dof_states["pos"][joint_idx] = target_qpos[i]
            gym.set_actor_dof_position_targets(env, hand_actor_handle,
                                               dof_states["pos"])

            hand_shape_props = gym.get_actor_rigid_shape_properties(
                env, hand_actor_handle)
            hand_rigid_body_set = set()
            for i in range(
                    gym.get_actor_rigid_body_count(env, hand_actor_handle)):
                hand_rigid_body_set.add(
                    gym.get_actor_rigid_body_index(env, hand_actor_handle, i,
                                                   gymapi.DOMAIN_ENV))
            self.hand_rigid_body_sets.append(hand_rigid_body_set)
            for i in range(len(hand_shape_props)):
                hand_shape_props[i].friction = self.hand_friction
            gym.set_actor_rigid_shape_properties(env, hand_actor_handle,
                                                 hand_shape_props)

            obj_rigid_body_sets = []
            for i in range(len(self.obj_asset_list)):
                pose = gymapi.Transform()
                pose.r = gymapi.Quat(*object_rotation[i][1:], object_rotation[i][0])
                pose.p = gymapi.Vec3(*object_translation[i])
            
                pose = test_rot * pose
                obj_actor_handle = gym.create_actor(
                    env, self.obj_asset_list[i], pose, f"obj_{i}", 0, 1)
                # import pdb; pdb.set_trace()
                self.obj_handles.append(obj_actor_handle)
                self.obj_init_poses.append(pose)
                gym.set_actor_scale(env, obj_actor_handle, obj_scales[i])
                obj_shape_props = gym.get_actor_rigid_shape_properties(
                    env, obj_actor_handle)
                obj_rigid_body_set = set()
                
                # import pdb; pdb.set_trace()
                for i in range(gym.get_actor_rigid_body_count(env, obj_actor_handle)):
                    
                    obj_rigid_body_set.add(gym.get_actor_rigid_body_index(env, obj_actor_handle, i,
                                                    gymapi.DOMAIN_ENV))
                obj_rigid_body_sets.append(obj_rigid_body_set)
                for i in range(len(obj_shape_props)):
                    obj_shape_props[i].friction = self.obj_friction
                gym.set_actor_rigid_shape_properties(env, obj_actor_handle,
                                                    obj_shape_props)
            self.obj_rigid_body_sets.append(obj_rigid_body_sets)


    def run_sim(self):
        for _ in range(self.sim_step):
            gym.simulate(self.sim)
            if self.has_viewer:
                sleep(self.debug_interval)
                if gym.query_viewer_has_closed(self.viewer):
                    break
                gym.step_graphics(self.sim)
                gym.draw_viewer(self.viewer, self.sim, False)

        success = []
        for i, env in enumerate(self.envs):
            contacts = gym.get_env_rigid_contacts(env)
            flag_per_obj = [0 for _ in range(len(self.obj_asset_list))]
            flag=False
            # import pdb;pdb.set_trace()
            for contact in contacts:
                # if (contact[2] in self.hand_rigid_body_sets[i]) and (
                #         contact[3] in self.obj_rigid_body_sets[i]):
                #     flag = True
                #     break
                # if (contact[3] in self.hand_rigid_body_sets[i]) and (
                #         contact[2] in self.obj_rigid_body_sets[i]):
                #     flag = True
                #     break
                
                if contact[2] in self.hand_rigid_body_sets[i]:
                    for j,sub_obj_set in enumerate(self.obj_rigid_body_sets[i]):
                        if contact[3] in sub_obj_set:
                            flag_per_obj[j] = 1
                    # flag = True
                    # break
                if contact[3] in self.hand_rigid_body_sets[i]:
                    for j,sub_obj_set in enumerate(self.obj_rigid_body_sets[i]):
                        if contact[2] in sub_obj_set:
                            flag_per_obj[j] = 1
                            
                    # flag = True
                    # break
                # if all objects has a contact
                if sum(flag_per_obj) == len(self.obj_asset_list):
                    flag = True
                    break
                
            success.append(flag)
        return success

    def reset_simulator(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)
            self.viewer = gym.create_viewer(self.sim, self.camera_props)
        self.sim = gym.create_sim(self.gpu, self.gpu, gymapi.SIM_PHYSX,
                                  self.sim_params)
        for env in self.envs:
            gym.destroy_env(env)
        self.envs = []
        self.hand_handles = []
        self.obj_handles = []
        self.hand_rigid_body_sets = []
        self.obj_rigid_body_sets = []
        self.hand_asset = None
        self.obj_asset = None
        self.obj_asset_list = []
        self.obj_init_poses = []

    def destroy(self):
        gym.destroy_sim(self.sim)
        if self.has_viewer:
            gym.destroy_viewer(self.viewer)