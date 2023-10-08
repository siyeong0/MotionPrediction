from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from mtss_gym.envs.base_task import BaseTask
from mtss_gym.envs.mtss_cfg import MtssCfg

class MotionTrackingFromSparseSensor(BaseTask):
    def __init__(self, cfg: MtssCfg, physics_engine, sim_device, headless):
        # parse config
        self.cfg = cfg
        self.obs_scales = self.cfg.normalization.obs_scales
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.dt = cfg.sim.dt
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # parse simulation params
        sim_params = gymapi.SimParams()
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

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.pre_physics_step()
        # TODO: reflect control frequency
        self.gym.simulate(self.sim)
        self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def pre_physics_step(self):
        forces = self.actions * self.motor_efforts.unsqueeze(0)
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

    def check_termination(self):
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
    
    def compute_reward(self):
        self.rew_buf[:] = 0.
        if "imitation" in self.cfg.reward.functions:
            self.rew_buf += self._reward_imitation()
        if "contact" in self.cfg.reward.functions:
            self.rew_buf += self._reward_contact()
        if "regularization" in self.cfg.reward.functions:
            self.rew_buf += self._reward_regularization()
    
    def compute_observations(self):
        # simulated character observation 
        dof_pos = torch.flatten(self.dof_pos, 1) * self.obs_scales.dof_pos
        dof_vel = torch.flatten(self.dof_vel, 1) * self.obs_scales.dof_vel
        body_pos = torch.flatten(self.body_pos, 1) * self.obs_scales.body_pos
        body_vel = torch.flatten(self.body_vel, 1) * self.obs_scales.body_vel
        body_quat = torch.flatten(self.body_quat, 1) * self.obs_scales.body_quat
        body_ang_vel = torch.flatten(self.body_ang_vel, 1) * self.obs_scales.body_ang_vel
        contact_force = torch.flatten(self.contact_force, 1) * self.obs_scales.contact_force
        obs_sim = torch.cat((dof_pos,dof_vel,body_pos,body_vel,body_quat,body_ang_vel,contact_force), dim=-1)
        # sensor position observation
        self.sensor_obs_hist[..., 1:self.num_stack-1] = self.sensor_obs_hist[..., 0:self.num_stack-2]
        sensor_pos = torch.flatten(self.body_pos[...,self.sensor_indices,:], 1) * self.obs_scales.body_pos
        headset_quat = torch.flatten(self.body_quat[...,self.sensor_indices[0],:], 1) * self.obs_scales.body_quat
        self.sensor_obs_hist[..., 0,:] = torch.cat((sensor_pos, headset_quat),dim=1)
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

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = 0
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = torch.clone(self.init_root_state)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

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
        self.base_quat = self.root_states[:, 3:7]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.body_state = gymtorch.wrap_tensor(body_state_tensor)
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,3:7]
        self.body_vel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,7:10]
        self.body_ang_vel = self.body_state.view(self.num_envs, self.num_bodies, 13)[...,10:13]
        self.contact_force = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.motor_efforts = to_torch(self.motor_efforts, device=self.device)
        
        # initialize some data used later on
        self.init_root_state = torch.clone(self.root_states)
        self.common_step_counter = 0
        self.extras = {}
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        self.num_stack = 6
        self.sensor_obs_hist = torch.zeros(self.num_envs, self.num_stack, (3+4+3+3), device=self.device)

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
        
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        actuator_props = self.gym.get_asset_actuator_properties(robot_asset)
        self.motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
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

    #------------ reward functions----------------
    def _reward_imitation(self):
        return torch.ones(self.num_envs)
    
    def _reward_contact(self):
        return torch.ones(self.num_envs)

    def _reward_regularization(self):
        return torch.ones(self.num_envs)
    
    #------------ helper functions----------------
    def sample_action(self):
        return (torch.rand(self.num_envs, self.num_actions) - 0.5) * 5
    