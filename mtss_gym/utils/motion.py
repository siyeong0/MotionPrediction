import numpy as np
import torch

from mtss_gym.utils.motion_lib import MotionLib

class Motion:
    def __init__(self, files, num_envs, dt, start_limit, min_length, num_dofs, num_bodies, device):
        self.num_envs = num_envs
        self.num_motions = len(files)
        self.dt = dt
        self.frame_stride = int(1 / self.dt)
        self.start_limit = start_limit
        self.min_length = min_length
        self.device = device
        
        self._motion_idxs = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self._motion_offsets = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        
        self._load(files, num_dofs, num_bodies)
        
    def _load(self, files, num_dofs, num_bodies):
        self.root_pos_tensors = []
        self.root_rot_tensors = []
        self.root_vel_tensors = []
        self.root_ang_vel_tensors = []
        self.dof_pos_tensors = []
        self.dof_vel_tensors = []
        self.key_pos_tensors = []
        self.key_vel_tensors = []
        self.key_rot_tensors = []
        
        motion_lengths = []
        for file in files:
            motion_file = file if "." in file else f"{file}.npy" 
            motion_lib = MotionLib(motion_file=motion_file,
                                   num_dofs=num_dofs,
                                   key_body_ids=np.array([x for x in range(num_bodies)]),
                                   device=self.device)
            # import to tensor
            num_frames = int(motion_lib.get_motion_length(0) / self.dt)
            motion_lengths.append(num_frames)
            # buffers
            root_pos_tensors = []
            root_rot_tensors = []
            root_vel_tensors = []
            root_ang_vel_tensors = []
            dof_pos_tensors = []
            dof_vel_tensors = []
            key_pos_tensors = []
            key_vel_tensors = []
            key_rot_tensors = []
            # to torch tensor
            for n in range(num_frames):
                state = motion_lib.get_motion_state(np.array([0]), np.array([self.dt * n]))
                root_pos, root_rot ,root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot = state
                root_pos_tensors.append(root_pos[0])
                root_rot_tensors.append(root_rot[0])
                root_vel_tensors.append(root_vel[0])
                root_ang_vel_tensors.append(root_ang_vel[0])
                dof_pos_tensors.append(dof_pos[0])
                dof_vel_tensors.append(dof_vel[0])
                key_pos_tensors.append(key_pos[0])
                key_vel_tensors.append(key_vel[0])
                key_rot_tensors.append(key_rot[0])
            # append to buffer list
            self.root_pos_tensors.append(torch.stack(root_pos_tensors, dim=0))
            self.root_rot_tensors.append(torch.stack(root_rot_tensors, dim=0))
            self.root_vel_tensors.append(torch.stack(root_vel_tensors, dim=0))
            self.root_ang_vel_tensors.append(torch.stack(root_ang_vel_tensors, dim=0))
            self.dof_pos_tensors.append(torch.stack(dof_pos_tensors, dim=0))
            self.dof_vel_tensors.append(torch.stack(dof_vel_tensors, dim=0))
            self.key_pos_tensors.append(torch.stack(key_pos_tensors, dim=0))
            self.key_vel_tensors.append(torch.stack(key_vel_tensors, dim=0))
            self.key_rot_tensors.append(torch.stack(key_rot_tensors, dim=0))
        self._motion_lengths = torch.tensor(motion_lengths, device=self.device, dtype=torch.long)
        
        print("===================================================")
        print("Loaded {:s} files with a total length of {:.3f}" \
            .format(file, sum([length * self.dt for length in motion_lengths]))) # assert one motion_lib contain only one motion data
        print("===================================================")
        
    def reset(self, env_ids):
        self._motion_idxs[env_ids] = torch.randint(0, self.num_motions, (len(env_ids),), device=self.device)
        self._motion_offsets[env_ids] = torch.zeros(env_ids.shape[0], device=self.device, dtype=torch.long)
            
    def step_motion_state(self, env_ids):
        state = self.get_motion_state(env_ids)
        self._motion_offsets[env_ids] += 1
        done = self._motion_offsets[env_ids] >= self._motion_lengths[self._motion_idxs[env_ids]]
        
        return state, done
    
    def get_motion_state(self, env_ids):
        root_pos = torch.zeros(env_ids.shape[0], 3, device=self.device, dtype=torch.float32)
        root_rot = torch.zeros(env_ids.shape[0], 4, device=self.device, dtype=torch.float32)
        root_vel = torch.zeros(env_ids.shape[0], 3, device=self.device, dtype=torch.float32)
        root_ang_vel = torch.zeros(env_ids.shape[0], 3, device=self.device, dtype=torch.float32)
        dof_pos = torch.zeros(env_ids.shape[0], 28, device=self.device, dtype=torch.float32)
        dof_vel = torch.zeros(env_ids.shape[0], 28, device=self.device, dtype=torch.float32)
        key_pos = torch.zeros(env_ids.shape[0], 15, 3, device=self.device, dtype=torch.float32)
        key_vel = torch.zeros(env_ids.shape[0], 15, 3, device=self.device, dtype=torch.float32)
        key_rot = torch.zeros(env_ids.shape[0], 15, 4, device=self.device, dtype=torch.float32)
        # TODO:
        
        for i, env_id in enumerate(env_ids):
            motion_idx = self._motion_idxs[env_id]
            root_pos[i] = self.root_pos_tensors[motion_idx][self._motion_offsets[env_id]]
            root_rot[i] = self.root_rot_tensors[motion_idx][self._motion_offsets[env_id]]
            root_vel[i] = self.root_vel_tensors[motion_idx][self._motion_offsets[env_id]]
            root_ang_vel[i] = self.root_ang_vel_tensors[motion_idx][self._motion_offsets[env_id]]
            dof_pos[i] = self.dof_pos_tensors[motion_idx][self._motion_offsets[env_id]]
            dof_vel[i] = self.dof_vel_tensors[motion_idx][self._motion_offsets[env_id]]
            key_pos[i] = self.key_pos_tensors[motion_idx][self._motion_offsets[env_id]]
            key_vel[i] = self.key_vel_tensors[motion_idx][self._motion_offsets[env_id]]
            key_rot[i] = self.key_rot_tensors[motion_idx][self._motion_offsets[env_id]]
            
        return (root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot)