import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, calc_heading_quat, calc_heading_quat_inv, quaternion_to_matrix
from utils.torch_jit_utils import exp_neg_norm_square, quat_to_nn_rep, rot_mat_to_vec
from utils.motion import Motion
from mtss_gym.envs.base_task import BaseTask
from mtss_cfg.mtss_cfg import MtssCfg

class MotionTrackingFromSparseSensor(BaseTask):
    def __init__(self, cfg: MtssCfg, physics_engine, sim_device, headless):
        # parse config
        self.cfg = cfg
        self.num_past_frame = self.cfg.env.num_past_frame
        self.num_future_frame = self.cfg.env.num_future_frame
        self.num_frames = self.num_past_frame + 1 + self.num_future_frame
        self.time_stride = self.cfg.env.time_stride
        self.max_episode_length_s = self.cfg.env.max_episode_length_s
        self.dt = cfg.sim.dt
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
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
        self._load_motions()
        self._init_buffers()
        self._normalize_coefs()
        
    def play(self):
        while True:
            # play randomly selected motion
            env_ids = self._every_env_ids()
            self._set_env_state(env_ids)
            # simulate
            self.gym.simulate(self.sim)
            self.render()
            # refresh actor states and steps additional works
            self.post_physics_step()
        
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.pre_physics_step()
        self.gym.simulate(self.sim)
        if not self.headless:
            self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
            
        self.prev_actions = self.actions
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

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

        # motion step
        done = self.motion.step_motion_state()
        self.motion_end_buf = to_torch(done, dtype=torch.bool, device=self.device)

        # compute reward before to reset and update obs_buf
        self.compute_reward()
        
        # reset environments
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        
        # compute observation
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

    def check_termination(self):
        motion_state = self.motion.get_motion_state()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.fall_down_buf = torch.logical_or(torch.abs(self.root_pos[:,2] - motion_state.root_pos[:,2]) > 1.0, torch.abs(self.link_pos[:,2,2] - motion_state.link_pos[:,2,2]) > 1.0)
        self.reset_buf = torch.logical_or(torch.logical_or(self.time_out_buf, self.motion_end_buf), self.fall_down_buf)

        #self.rew_buf -= self.fall_down_buf.float() * 10.

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # reset motions
        self.motion.reset(env_ids)
        
        # initialize motion state buffer and set actor root and dof state
        self._init_env_state(env_ids)

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    
    def compute_observations(self):
        # simulated character observation 
        obs_sim = compute_sim_observation(self.root_state, self.dof_pos, self.dof_vel, 
                                          self.link_pos, self.link_quat, self.link_vel, self.link_ang_vel, 
                                          self.contact_force)
        
        # user sensor position observation
        obs_user_buf = []
        curr_motion_state = self.motion.get_motion_state(0.0)
        obs_user_buf.append(compute_user_observation(self.root_state,
                                                     curr_motion_state.link_pos, curr_motion_state.link_rot, 
                                                     self.pos_sensor_indices))
        for i in range(self.num_past_frame):
            past_motion_state = self.motion.get_motion_state(-(i+1) * self.time_stride)
            obs_user_buf.insert(0, compute_user_observation(self.root_state, 
                                                            past_motion_state.link_pos, past_motion_state.link_rot, 
                                                            self.pos_sensor_indices))
        for i in range(self.num_future_frame):
            future_motion_state = self.motion.get_motion_state((i+1) * self.time_stride)
            obs_user_buf.append(compute_user_observation(self.root_state, 
                                                         future_motion_state.link_pos, future_motion_state.link_rot, 
                                                         self.pos_sensor_indices))
        obs_user = torch.cat(obs_user_buf, dim=1)
        # height map observation
        # TODO:
        
        # concat
        self.obs_buf = torch.cat((obs_sim, obs_user), dim=1)
        
    def compute_reward(self):
        coef = self.cfg.reward.coef
        self.rew_buf[:] = 0.
        if "imitation" in self.cfg.reward.functions:
            motion_state = self.motion.get_motion_state()
            self.rew_buf += coef.w_i \
            * compute_imitation_reward(self.root_state, self.dof_state, self.link_state,
                                       motion_state.root_state, motion_state.dof_state, motion_state.link_state,
                                       to_torch(coef.imitation.w_p), to_torch(coef.imitation.w_pv), to_torch(coef.imitation.w_q), to_torch(coef.imitation.w_qv), to_torch(coef.imitation.w_r),
                                       to_torch(coef.imitation.k_p), to_torch(coef.imitation.k_pv), to_torch(coef.imitation.k_q), to_torch(coef.imitation.k_qv), to_torch(coef.imitation.k_r))
        if "contact" in self.cfg.reward.functions:
            pass
            # self.rew_buf += coef.w_c * compute_contact_reward()
        if "regularization" in self.cfg.reward.functions:
            self.rew_buf += coef.w_r * compute_regularization_reward(self.actions, self.prev_actions,
                                                                     to_torch(coef.regularization.w_a), to_torch(coef.regularization.w_s),
                                                                     to_torch(coef.regularization.k_a), to_torch(coef.regularization.k_s))

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()
        
        
    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

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
        
        self.motion_end_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.fall_down_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
    def _load_motions(self):
        self.motion_libs = []
        dir = self.cfg.motion.dir
        files = self.cfg.motion.files
        self.motion = Motion([f"{dir}/{file}" for file in files],
                             self.num_envs, 
                             self.dt,
                             self.num_past_frame * self.time_stride,
                             self.num_future_frame * self.time_stride,
                             self.num_dofs, self.num_bodies,
                             self.device)
    
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
            
    def _init_env_state(self, env_ids):
        # self.root_state[env_ids, :] = torch.clone(self.init_root_state[env_ids, :])
        # self.dof_state[env_ids, :, :] = torch.clone(self.init_dof_state[env_ids, :, :])
        
        # self.root_state[env_ids, 0:2] = self.motion.get_motion_state().root_state[env_ids, 0:2]
        # self.root_state[env_ids, 3:7] = calc_heading_quat(self.motion.get_motion_state().root_state[env_ids, 3:7])
        
        self.root_state[env_ids, :] = self.motion.get_motion_state().root_state[env_ids, :]
        self.dof_state[env_ids, :, :] = self.motion.get_motion_state().dof_state[env_ids, :, :]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
            
    def _set_env_state(self, env_ids):
        self.root_state[env_ids, :] = self.motion.get_motion_state().root_state[env_ids,:]
        self.dof_state[env_ids, :, :] = self.motion.get_motion_state().dof_state[env_ids,:,:]
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(torch.clone(self.dof_pos)),
        #                                             gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
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

@torch.jit.script
def compute_user_observation(root_state, link_pos, link_quat, sensor_indices):
    key_pos = link_pos[:, sensor_indices]
    key_quat = link_quat[:, sensor_indices[0]]  # first element of sensor_indices is headset index
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

@torch.jit.script
def compute_imitation_reward(sim_root_state, sim_dof_state, sim_link_state, ref_root_state, ref_dof_state, ref_link_state, 
                             w_p, w_pv, w_q, w_qv, w_r, k_p, k_pv, k_q, k_qv, k_r):
    num_envs = sim_root_state.shape[0]
    num_dofs = sim_dof_state.shape[1]
    num_links = sim_link_state.shape[1]
    # base_pos is pos that projects the pervis pos on ground
    sim_base_pos = torch.clone(sim_root_state[:, 0:3])
    sim_base_pos[:,2] = 0
    # base_quat is quaternion represents avatar's heading; yaw
    sim_inv_base_quat = calc_heading_quat_inv(sim_root_state[:,3:7])
    # preproces inputs. prefix c means variable for calculating
    sim_c_base_pos = sim_base_pos.unsqueeze(1).repeat(1, num_links, 1).view(-1, 3)    # (1, num_links, 1)
    sim_c_inv_base_quat = sim_inv_base_quat.unsqueeze(1).repeat(1, num_links, 1).view(-1, 4)
    sim_dof_pos = sim_dof_state[:,:,0:1]
    sim_dof_vel = sim_dof_state[:,:,1:2]
    sim_c_link_pos = sim_link_state[:,:,0:3].view(-1,3)
    sim_c_link_quat = sim_link_state[:,:,3:7].view(-1,4)
    sim_c_link_vel = sim_link_state[:,:,7:10].view(-1,3)
    ref_dof_pos = ref_dof_state[:,:,0:1]
    ref_dof_vel = ref_dof_state[:,:,1:2]
    ref_c_link_pos = ref_link_state[:,:,0:3].view(-1,3)
    ref_c_link_quat = ref_link_state[:,:,3:7].view(-1,4)
    ref_c_link_vel = ref_link_state[:,:,7:10].view(-1,3)

    # simulation motions
    sim_dof_pos          = sim_dof_pos # not affected
    sim_dof_vel          = sim_dof_vel # not affected
    sim_link_pos         = quat_rotate(sim_c_inv_base_quat, (sim_c_link_pos-sim_c_base_pos)).view(num_envs, num_links, 3)
    sim_link_vel         = quat_rotate(sim_c_inv_base_quat, sim_c_link_vel).view(num_envs, num_links, 3)
    sim_link_mat_inv     = torch.transpose(quaternion_to_matrix(quat_mul(sim_c_link_quat, sim_c_inv_base_quat)), 1, 2) # quat_mul(target, transform)

    # reference motions
    ref_dof_pos          = ref_dof_pos # not affected
    ref_dof_vel          = ref_dof_vel # not affected
    ref_link_pos         = quat_rotate(sim_c_inv_base_quat, (ref_c_link_pos-sim_c_base_pos)).view(num_envs, num_links, 3)
    ref_link_vel         = quat_rotate(sim_c_inv_base_quat, ref_c_link_vel).view(num_envs, num_links, 3)
    ref_link_mat         = quaternion_to_matrix(quat_mul(ref_c_link_quat, sim_c_inv_base_quat)) # quat_mul(target, transform)
    
    # compute
    r_im_dof_pos    = torch.mean(exp_neg_norm_square(k_p, (sim_dof_pos-ref_dof_pos)).view(num_envs, -1), 1)
    r_im_dof_vel    = torch.mean(exp_neg_norm_square(k_pv, (sim_dof_vel-ref_dof_vel)).view(num_envs, -1), 1)
    r_im_link_pos   = torch.mean(exp_neg_norm_square(k_q, (sim_link_pos-ref_link_pos)).view(num_envs, -1), 1)
    r_im_link_vel   = torch.mean(exp_neg_norm_square(k_qv, (sim_link_vel-ref_link_vel)).view(num_envs, -1), 1)
    r_im_link_ori   = torch.mean(exp_neg_norm_square(k_r, rot_mat_to_vec(sim_link_mat_inv @ ref_link_mat)).view(num_envs, -1), 1)

    r_im = w_p * r_im_dof_pos \
        + w_pv * r_im_dof_vel \
        + w_q * r_im_link_pos \
        + w_qv * r_im_link_vel \
        + w_r * r_im_link_ori
    return r_im

@torch.jit.script
def compute_contact_reward():
    pass

@torch.jit.script
def compute_regularization_reward(actions, prev_actions, w_a, w_s, k_a, k_s):
    num_envs = actions.shape[0]
    
    r_reg_a = torch.mean(exp_neg_norm_square(k_a, actions).view(num_envs, -1), 1)
    r_reg_s = torch.mean(exp_neg_norm_square(k_s, (actions-prev_actions)).view(num_envs, -1), 1)
    
    r_reg = w_a * r_reg_a \
        + w_s * r_reg_s
    return r_reg