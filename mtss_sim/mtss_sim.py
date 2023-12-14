import numpy as np
import os

from rsl_rl.runners import OnPolicyRunner

from utils.helpers import class_to_dict, get_log_dir

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat, calc_heading_quat_inv, quaternion_to_matrix
from utils.torch_jit_utils import exp_neg_norm_square, quat_to_nn_rep, rot_mat_to_vec
from mtss_sim.base_task import BaseTask
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

class MotionTrackingSim(BaseTask):
    def __init__(self, 
                 cfg: MtssCfg, 
                 max_num_envs: int,
                 rl_cfg: MtssPPOCfg,
                 model_path: str,  
                 physics_engine, 
                 sim_device, 
                 headless):
        # parse config
        self.cfg = cfg
        self.cfg.env.num_envs = max_num_envs
        self.num_past_frame = self.cfg.env.num_past_frame
        self.num_future_frame = self.cfg.env.num_future_frame
        self.num_frames = self.num_past_frame + 1 + self.num_future_frame
        self.time_stride = self.cfg.env.time_stride
        self.dt = cfg.sim.dt
        # parse simulation params
        sim_params = gymapi.SimParams()
        sim_params.use_gpu_pipeline = cfg.sim.use_gpu
        sim_params.dt = cfg.sim.dt
        sim_params.substeps = cfg.sim.substeps
        sim_params.gravity = gymapi.Vec3(cfg.sim.gravity[0], cfg.sim.gravity[1], cfg.sim.gravity[2])
        sim_params.up_axis = gymapi.UP_AXIS_Y if cfg.sim.up_axis == 0 else gymapi.UP_AXIS_Z
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
        sim_params.physx.solver_type = cfg.sim.physx.solver_type
        sim_params.physx.num_position_iterations = cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = cfg.sim.physx.contact_offset
        sim_params.physx.rest_offset = cfg.sim.physx.rest_offset
        sim_params.physx.bounce_threshold_velocity = cfg.sim.physx.bounce_threshold_velocity
        sim_params.physx.max_depenetration_velocity = cfg.sim.physx.max_depenetration_velocity
        sim_params.physx.max_gpu_contact_pairs = cfg.sim.physx.max_gpu_contact_pairs
        sim_params.physx.default_buffer_size_multiplier = cfg.sim.physx.default_buffer_size_multiplier
        self.sim_params = sim_params
        # initialize base task
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        # set rendering camera
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        
        # initialize class buffers
        self._init_buffers()
        self._normalize_coefs()
        
        # load rl model
        self.rl_cfg = rl_cfg
        self.rl_cfg.runner.resume = True
        self.load_model(model_path)
        
    def step(self, hmd_state_stack, left_hand_pos_stack=None, right_hand_pos_stack=None):
        self.hmd_state_buffer[:] = hmd_state_stack
        self.hand_state_buffer[:, :, 0, :] = left_hand_pos_stack if left_hand_pos_stack != None else 0
        self.hand_state_buffer[:, :, 1, :] = right_hand_pos_stack if right_hand_pos_stack != None else 0
            
         # compute observation
        self.compute_observations()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        # get policy model output; action
        obs = self.get_observations()
        actions = self.policy(obs.detach()).detach()
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # simulate
        self.pre_physics_step()
        self.gym.simulate(self.sim)
        if not self.headless:
            self.render()
        self.post_physics_step()
            
        self.prev_actions = self.actions
        
        # return full body motion
        return self.root_state, self.dof_state

    def pre_physics_step(self):
        forces = self.actions * self.motor_efforts.unsqueeze(0)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        # refresh states
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        # reset environments
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
    
    def compute_observations(self):
        # simulated character observation 
        obs_sim = compute_sim_observation(self.root_state, self.dof_pos, self.dof_vel, 
                                          self.link_pos, self.link_quat, self.link_vel, self.link_ang_vel, 
                                          self.contact_force)
        
        # user sensor position observation
        obs_user_buf = []
        for i in range(1 + self.num_past_frame + self.num_future_frame):
            obs_user_buf.append(compute_user_observation(self.root_state,
                                                        self.hmd_state_buffer[i],
                                                        self.hand_state_buffer[i]))
        obs_user = torch.cat(obs_user_buf, dim=1)
        # height map observation
        # TODO:
        
        # concat
        self.obs_buf = torch.cat((obs_sim, obs_user), dim=1)
        
    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
    
    def init_env_state(self, env_ids, root_state):
        self.root_state[env_ids, :] = root_state[env_ids]
        self.dof_state[env_ids, :, :] = torch.clone(self.init_dof_state[env_ids, :, :])
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def load_model(self, path):
        ppo_runner = OnPolicyRunner(self, class_to_dict(MtssPPOCfg), get_log_dir("logs", "mtss"), device="cuda")
        ppo_runner.load(path)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

    #----------------------------------------
    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        # refresh states
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_state = gymtorch.wrap_tensor(actor_root_state)
        self.root_pos = self.root_state[:, 0:3]
        self.root_quat = self.root_state[:, 3:7]
        self.root_vel = self.root_state[:, 7:10]
        self.root_ang_vel = self.root_state[:, 10:13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.link_state = self.body_state.view(self.num_envs, self.num_bodies, 13)
        self.link_pos = self.link_state[...,0:3]
        self.link_quat = self.link_state[...,3:7]
        self.link_vel = self.link_state[...,7:10]
        self.link_ang_vel = self.link_state[...,10:13]
        self.contact_force = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.motor_efforts = to_torch(self.motor_efforts, device=self.device)
        
        # initialize some data used later on
        self.init_root_state = torch.zeros_like(self.root_state)
        self.init_root_state[:, 2:3] = 0.85
        self.init_root_state[:, 6:7] = 1.0
        self.init_dof_state = torch.zeros_like(self.dof_state)
        
        self.common_step_counter = 0
        self.extras = {}
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)
        
        num_stack = self.num_past_frame + self.num_future_frame + 1
        self.hmd_state_buffer = torch.zeros(num_stack, self.num_envs, 7, device=self.device, dtype=torch.float32)
        self.hand_state_buffer = torch.zeros(num_stack, self.num_envs, 2, 3, device=self.device, dtype=torch.float32)
    
    def _normalize_coefs(self):
        coef = self.cfg.reward.coef
        sum = coef.w_i + coef.w_c + coef.w_r
        coef.w_i /= sum
        coef.w_c /= sum
        coef.w_r /= sum
        
        coef = self.cfg.reward.coef.imitation
        sum = coef.w_q + coef.w_qv + coef.w_p + coef.w_pv + coef.w_r
        coef.w_q /= sum
        coef.w_qv /= sum
        coef.w_p /= sum
        coef.w_pv /= sum
        coef.w_r /= sum
        
        coef = self.cfg.reward.coef.contact
        sum = coef.w_c
        coef.w_c /= sum
        
        coef = self.cfg.reward.coef.regularization
        sum = coef.w_a + coef.w_s
        coef.w_a /= sum
        coef.w_s /= sum

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.dynamic_friction = 1.2
        plane_params.static_friction = 1.2
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        # load model asset
        asset_path = self.cfg.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        actuator_props = self.gym.get_asset_actuator_properties(robot_asset)
        self.motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        pos_sensor_names = [s for s in body_names if s in self.cfg.asset.pos_sensor_body_names]
        force_sensor_names = [s for s in body_names if s in self.cfg.asset.force_sensor_body_names]
        # create envs
        spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.85)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        # sensor indices tensors
        self.pos_sensor_indices = torch.zeros(len(pos_sensor_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(pos_sensor_names)):
            self.pos_sensor_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], pos_sensor_names[i])
        self.force_sensor_indices = torch.zeros(len(force_sensor_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(force_sensor_names)):
            self.force_sensor_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], force_sensor_names[i])
            
    def _set_env_state(self, env_ids):
        self.root_state[env_ids, :] = self.motion.get_motion_state().root_state[env_ids,:]
        self.dof_state[env_ids, :, :] = self.motion.get_motion_state().dof_state[env_ids,:,:]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(torch.clone(self.dof_pos)),
        #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32hmd_state_buffer
    #------------ helper functions----------------
    def sample_action(self):
        action = (torch.rand(self.num_envs, self.num_actions) - 0.5) * 5.0
        return action

    def _every_env_ids(self):
        return to_torch([x for x in range(self.num_envs)], dtype=torch.long, device=self.device)
        
@torch.jit.script
def compute_sim_observation(root_state, dof_pos, dof_vel, link_pos, link_quat, link_vel, link_ang_vel, contact_force):
    num_envs = root_state.shape[0]
    # base_pos is pos that projects the pervis pos on ground
    base_pos = torch.clone(root_state[:, 0:3])
    base_pos[:,2] = 0
    # base_quat is quaternion represents avatar's heading; yaw
    inv_base_quat = calc_heading_quat_inv(root_state[:,3:7])
    # preproces inputs. prefix c means variable for calculating
    c_base_pos = base_pos.unsqueeze(1).repeat(1, link_pos.shape[1], 1).view(-1, 3)    # (1, num_links, 1)
    c_inv_base_quat = inv_base_quat.unsqueeze(1).repeat(1, link_pos.shape[1], 1).view(-1, 4)
    c_link_pos = link_pos.view(-1,3)
    c_link_quat = link_quat.view(-1,4)
    c_link_vel = link_vel.view(-1,3)
    c_link_ang_vel = link_ang_vel.view(-1,3)
    
    # ac means avatarcentric
    # nn_rep means first and second columns of 3x3 rotation matrix
    # all tensors are flattened
    ac_dof_pos          = torch.flatten(dof_pos, 1) # not affected
    ac_dof_vel          = torch.flatten(dof_vel, 1) # not affected
    ac_link_pos         = torch.flatten(quat_rotate(c_inv_base_quat, (c_link_pos - c_base_pos)).view(num_envs, -1), 1)
    ac_link_ori         = torch.flatten(quat_to_nn_rep(quat_mul(c_link_quat, c_inv_base_quat)).view(num_envs, -1), 1) # quat_mul(target, transform)
    ac_link_vel         = torch.flatten(quat_rotate(c_inv_base_quat, c_link_vel).view(num_envs, -1), 1)
    ac_link_ang_vel     = torch.flatten(quat_rotate(c_inv_base_quat, c_link_ang_vel).view(num_envs, -1), 1)
    ac_contact_force    = torch.flatten(contact_force, 1) # not affected
    return torch.cat((ac_dof_pos, ac_dof_vel, 
                      ac_link_pos, ac_link_ori, 
                      ac_link_vel, ac_link_ang_vel, 
                      ac_contact_force), dim=1)

#@torch.jit.script
def compute_user_observation(root_state, hmd_state, hand_state):
    key_pos = torch.stack((hmd_state[:, 0:3], hand_state[:, 0, 0:3], hand_state[:, 1, 0:3]), dim=1)
    key_quat = hmd_state[:, 3:7]
    num_envs = root_state.shape[0]
    # base_pos is pos that projects the pervis pos on ground
    base_pos = torch.clone(root_state[:, 0:3])
    base_pos[:,2] = 0
    # base_quat is quaternion represents avatar's heading; yaw
    inv_base_quat = calc_heading_quat_inv(root_state[:,3:7])
    # preproces inputs. prefix c means variable for calculating
    c_base_pos = base_pos.unsqueeze(1).repeat(1, key_pos.shape[1], 1).view(-1, 3)    # (1, num_pos_sensors, 1)
    c_inv_base_quat = inv_base_quat.unsqueeze(1).repeat(1, key_pos.shape[1], 1).view(-1, 4)    
    c_inv_base_quat_ori = inv_base_quat # headset orientation only
    c_key_pos = key_pos.view(-1,3)
    c_key_quat = key_quat.view(-1,4)
    
    # ac means avatarcentric
    # nn_rep means first and second columns of 3x3 rotation matrix
    # all tensors are flattened
    ac_key_pos = torch.flatten(quat_rotate(c_inv_base_quat, (c_key_pos - c_base_pos)).view(num_envs, -1), 1)
    ac_key_ori = torch.flatten(quat_to_nn_rep(quat_mul(c_key_quat, c_inv_base_quat_ori)).view(num_envs, -1), 1) # quat_mul(target, transform)
    return torch.cat((ac_key_pos, ac_key_ori), dim=1)