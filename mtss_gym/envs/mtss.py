import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params, calc_heading_quat_inv, \
     exp_map_to_quat, quat_to_tan_norm, my_quat_rotate, calc_heading_quat_inv
from ..utils.motion import Motion

from mtss_gym.envs.base_task import BaseTask
from mtss_gym.envs.mtss_cfg import MtssCfg

class MotionTrackingFromSparseSensor(BaseTask):
    def __init__(self, cfg: MtssCfg, physics_engine, sim_device, headless):
        # parse config
        self.cfg = cfg
        self.num_stack = self.cfg.env.num_stack
        self.min_episode_length_s = self.cfg.env.min_episode_length_s
        self.max_episode_length_s = self.cfg.env.max_episode_length_s
        self.dt = cfg.sim.dt
        self.control_dt = cfg.sim.control_dt
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.control_dt)
        self.num_update = int(self.control_dt / self.dt)
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
        self._load_motions()
        
    def play(self):
        self.control_dt = self.dt
        self.num_update = 1
        self.motion.dt = self.dt
        while True:
            # play randomly selected motion
            env_ids = self._every_env_ids()
            state, done = self.motion.step_motion_state(env_ids)
            self.motion_end_buf[env_ids] = to_torch(done, dtype=torch.bool, device=self.device)
            root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot = state
            self._set_env_state(env_ids=env_ids, 
                                root_pos=root_pos, 
                                root_rot=root_rot,
                                root_vel=root_vel,
                                root_ang_vel=root_ang_vel,  
                                dof_pos=dof_pos,  
                                dof_vel=dof_vel)
            # simulate
            for _ in range(self.num_update):
                self.gym.simulate(self.sim)
                self.render()
            # refresh actor states and steps additional works
            self.post_physics_step()
        
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.pre_physics_step()
        for _ in range(self.num_update):
            self.gym.simulate(self.sim)
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
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # motion step
        env_ids = self._every_env_ids()
        state, done = self.motion.step_motion_state(env_ids)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot = state
        self.motion_root_state_buf[env_ids, 0:3] = root_pos
        self.motion_root_state_buf[env_ids, 3:7] = root_rot
        self.motion_root_state_buf[env_ids, 7:10] = root_vel
        self.motion_root_state_buf[env_ids, 10:13] = root_ang_vel
        self.motion_dof_state_buf[env_ids, :, 0] = dof_pos
        self.motion_dof_state_buf[env_ids, :, 1] = dof_vel
        self.motion_link_state_buf[env_ids, :, 0:3] = key_pos
        self.motion_link_state_buf[env_ids, :, 3:6] = key_vel
        self.motion_link_state_buf[env_ids, :, 6:10] = key_rot
        
        self.motion_end_buf[env_ids] = to_torch(done, dtype=torch.bool, device=self.device)

        # compute observations, rewards, resets, ...
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_reward()
        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = torch.logical_or(self.time_out_buf, self.motion_end_buf)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_actors(env_ids)

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    
    def compute_reward(self):
        coef = self.cfg.reward.coef
        self.rew_buf[:] = 0.
        if "imitation" in self.cfg.reward.functions:
            self.rew_buf += coef.w_i * self._reward_imitation()
        if "contact" in self.cfg.reward.functions:
            self.rew_buf += coef.w_c * self._reward_contact()
        if "regularization" in self.cfg.reward.functions:
            self.rew_buf += coef.w_r * self._reward_regularization()
    
    def compute_observations(self):
        # simulated character observation 
        self.obs_dof_pos = torch.flatten(self.dof_pos, 1)
        self.obs_dof_vel = torch.flatten(self.dof_vel, 1)
        self.obs_body_pos = self._to_avatar_centric_coord_p(self.glob_body_pos, self.root_states)
        self.obs_body_vel = self._to_avatar_centric_coord_v(self.glob_body_vel, self.root_states)
        self.obs_body_quat = self._to_avatar_centric_coord_r(self.glob_body_quat, self.root_states)
        self.obs_body_ang_vel = self._to_avatar_centric_coord_v(self.glob_body_ang_vel, self.root_states)
        self.obs_contact_force = torch.flatten(self.contact_force, 1)
        
        obs_sim = torch.cat((self.obs_dof_pos,self.obs_dof_vel,
                             self.obs_body_pos,self.obs_body_vel,
                             self.obs_body_quat,self.obs_body_ang_vel,
                             self.obs_contact_force), dim=-1)
        # sensor position observation
        sensor_pos = self._to_avatar_centric_coord_p(self.motion_link_state_buf[:, self.sensor_indices, 0:3], self.motion_root_state_buf)
        headset_quat = self._to_avatar_centric_coord_r(self.motion_link_state_buf[:, self.sensor_indices[0]:self.sensor_indices[0]+1, 6:10], self.motion_root_state_buf)
        if self.num_stack > 1:
            self.sensor_obs_hist[:, 1:self.num_stack-1] = self.sensor_obs_hist[:, 0:self.num_stack-2]
        self.sensor_obs_hist[:, 0,:] = torch.cat((sensor_pos, headset_quat),dim=1)
        obs_user = torch.flatten(self.sensor_obs_hist, 1)
        
        # concat
        self.obs_buf = torch.cat((obs_sim, obs_user), dim=-1)

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _reset_actors(self, env_ids):
        self.motion.reset(env_ids)

        state = self.motion.get_motion_state(env_ids)
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot = state
        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,  
                            dof_pos=dof_pos,  
                            dof_vel=dof_vel)
        self.sensor_obs_hist[env_ids] = 0

    #----------------------------------------
    def _init_buffers(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_pos = self.root_states[:, 0:3]
        self.root_quat = self.root_states[:, 3:7]
        self.root_vel = self.root_states[:, 7:10]
        self.root_ang_vel = self.root_states[:, 10:13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.glob_body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,0:3]
        self.glob_body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,3:7]
        self.glob_body_vel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,7:10]
        self.glob_body_ang_vel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,10:13]
        self.contact_force = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.motor_efforts = to_torch(self.motor_efforts, device=self.device)
        
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device, dtype=torch.float32)
        self.prev_actions = torch.zeros_like(self.actions)
        
        self.motion_root_state_buf = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float32)
        self.motion_dof_state_buf = torch.zeros(self.num_envs, self.num_dofs, 2, device=self.device, dtype=torch.float32)
        self.motion_link_state_buf = torch.zeros(self.num_envs, self.num_bodies, 10, device=self.device, dtype=torch.float32)
        
        self.motion_end_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.sensor_obs_hist = torch.zeros(self.num_envs, self.num_stack, (3+4+3+3), device=self.device)
        
    def _load_motions(self):
        self.motion_libs = []
        dir = self.cfg.motion.dir
        files = self.cfg.motion.files
        self.motion = Motion([f"{dir}/{file}" for file in files],
                             self.num_envs, 
                             self.control_dt, 0.0,
                             self.min_episode_length_s,
                             self.num_dofs, self.num_bodies,
                             self.device)
        self._reset_actors(self._every_env_ids())

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
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
        sensor_names = [s for s in body_names if s in self.cfg.asset.sensor_body_name]

        # create envs
        spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.actor_handles = []
        self.envs = []
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.89)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
                
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.sensor_indices = torch.zeros(len(sensor_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(sensor_names)):
            self.sensor_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], sensor_names[i])
            
    def _set_env_state(self, env_ids, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel):
        self.root_states[env_ids, :] = 0
        self.root_states[env_ids, 0:3] = root_pos
        self.root_states[env_ids, 3:7] = root_rot
        #self.root_states[env_ids, 7:10] = root_vel
        #self.root_states[env_ids, 10:13] = root_ang_vel
        
        self.dof_state[env_ids, :] = 0
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), 
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    #------------ reward functions----------------
    def _reward_imitation(self):
        coef = self.cfg.reward.coef.imitation
        ref_dof_pos = torch.flatten(self.motion_dof_state_buf[:,:,0], 1)
        ref_dof_vel = torch.flatten(self.motion_dof_state_buf[:,:,1], 1)
        ref_body_pos = self._to_avatar_centric_coord_p(self.motion_link_state_buf[:,:,0:3], self.motion_root_state_buf)
        ref_body_vel = self._to_avatar_centric_coord_v(self.motion_link_state_buf[:,:,3:6], self.motion_root_state_buf)
        ref_body_rot = self._to_avatar_centric_coord_r(self.motion_link_state_buf[:,:,6:10], self.motion_root_state_buf)
        
        r_dof_pos_im = coef.w_p * _exp_neg_square(coef.k_p, (self.obs_dof_pos-ref_dof_pos))
        r_dof_vel_im = coef.w_pv * _exp_neg_square(coef.k_pv, (self.obs_dof_vel-ref_dof_vel))
        r_link_pos_im = coef.w_q * _exp_neg_square(coef.k_q, (self.obs_body_pos-ref_body_pos))
        r_link_vel_im = coef.w_qv * _exp_neg_square(coef.k_qv, (self.obs_body_vel-ref_body_vel))
        r_link_rot_im = coef.w_r * _exp_neg_square(coef.k_r, (self.obs_body_quat-ref_body_rot))
        
        r_im = r_dof_pos_im + r_dof_vel_im \
            + r_link_pos_im + r_link_vel_im \
            + r_link_rot_im
        return r_im / 5.
    
    def _reward_contact(self):
        coef = self.cfg.reward.coef.contact
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_regularization(self):
        coef = self.cfg.reward.coef.regularization
        r_a_reg = coef.w_a * _exp_neg_square(coef.q_a, self.actions)
        r_s_reg = coef.w_s * _exp_neg_square(coef.q_s, (self.actions-self.prev_actions))
        
        r_reg = r_a_reg + r_s_reg
        return r_reg / 2.
    
    #------------ helper functions----------------
    def sample_action(self):
        action = (torch.rand(self.num_envs, self.num_actions) - 0.5) * 5.0
        return action

    def _every_env_ids(self):
        return to_torch([x for x in range(self.num_envs)], dtype=torch.long, device=self.device)
    
    def _to_avatar_centric_coord_p(self, pos, root_state):
        base_pos = torch.clone(root_state[:, 0:3])
        base_pos[:,2] = 0
        base_pos_expand = base_pos.unsqueeze(-2)
        base_pos_expand = base_pos_expand.repeat((1, pos.shape[1], 1))  # TODO: repeat을 하지 않아도 결과가 같음.
        relative_pos = pos - base_pos_expand
        
        return self._to_avatar_centric_coord_v(relative_pos, root_state)
    
    def _to_avatar_centric_coord_v(self, vec, root_state):
        heading_rot = calc_heading_quat_inv(root_state[:, 3:7])
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, vec.shape[1], 1))
        
        flat_vec = vec.view(vec.shape[0] * vec.shape[1], vec.shape[2])
        flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], heading_rot_expand.shape[2])
        local_vec = my_quat_rotate(flat_heading_rot, flat_vec)
        flat_local_vec = local_vec.view(vec.shape[0], vec.shape[1] * vec.shape[2])
        
        return flat_local_vec
    
    def _to_avatar_centric_coord_r(self, quat, root_state):
        inv_root_quat = -root_state[:, 3:7]
        inv_root_quat[:,-1] *= -1. 
        inv_root_quat_expand = inv_root_quat.unsqueeze(-2)
        inv_root_quat_expand = inv_root_quat_expand.repeat((1, quat.shape[1], 1))
        
        flat_quat = quat.view(quat.shape[0] * quat.shape[1], quat.shape[2])
        flat_inv_root_quat = inv_root_quat_expand.view(inv_root_quat_expand.shape[0] * inv_root_quat_expand.shape[1], inv_root_quat_expand.shape[2])
        local_quat = quat_mul(flat_inv_root_quat, flat_quat)
        flat_local_quat = local_quat.view(quat.shape[0], quat.shape[1] * quat.shape[2])
        
        return flat_local_quat
    
@torch.jit.script
def _exp_neg_square(coef: float, tensor: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.exp(-coef * (torch.pow(tensor, 2))), 1) / tensor.shape[1]