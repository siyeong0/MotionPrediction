import numpy as np
import torch

from mtss_gym.utils.motion_lib import MotionLib
from mtss_gym.utils.motion_state import MotionState

class Motion:
    def __init__(self, files, num_envs, dt, min_past_time, min_future_time, num_dofs, num_bodies, device):
        self.num_envs = num_envs
        self.num_motions = len(files)
        self.dt = dt
        self.min_future_frame = min_future_time / self.dt
        self.min_past_frame = min_past_time / self.dt
        self.min_length_sec = min_past_time + min_future_time
        self.device = device
        self.num_dofs = num_dofs
        self.num_bodies = num_bodies
        
        self._motion_idxs = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        self._motion_offsets = torch.zeros(self.num_envs, device=device, dtype=torch.long)
        
        self._motion_state_tensors = []
        self._load(files)
        
    def _load(self, files):
        motion_lengths = []
        for i, file in enumerate(files):
            motion_file = file if "." in file else f"{file}.npy" 
            motion_lib = MotionLib(motion_file=motion_file,
                                   num_dofs=self.num_dofs,
                                   key_body_ids=np.array([x for x in range(self.num_bodies)]),
                                   device=self.device)
            # import to tensor
            num_frames = int(motion_lib.get_motion_length(0) / self.dt)
            # drop short data
            if self.min_length_sec * 2. > num_frames * self.dt:
                print("{:s} was droped; Length is under {:3f}s.".format(file, self.min_length_sec * 2.))
                self.num_motions -= 1
                continue
            # buffers
            motion_state_buffer = []
            # to torch tensor
            for n in range(num_frames):
                state = motion_lib.get_motion_state(np.array([0]), np.array([self.dt * n]))
                root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, key_vel, key_rot = state
                # TODO:
                root_pos[:,2] -= 0.05
                key_pos[:,:,2] -= 0.05
                motion_state_buffer.append(torch.cat((root_pos[0], root_rot[0], root_vel[0], root_ang_vel[0],
                                                      torch.flatten(torch.cat((dof_pos[0].unsqueeze(1), dof_vel[0].unsqueeze(1)), dim=1)), 
                                                      torch.flatten(key_pos[0]), torch.flatten(key_vel[0]), torch.flatten(key_rot[0])), dim=0))
            self._motion_state_tensors.append(torch.stack(motion_state_buffer, dim=0))
            motion_lengths.append(num_frames)
            # print progress
            num_1_per = int(self.num_motions / 100)
            if num_1_per == 0 or ((i+1) % num_1_per) == 0:
                print("{:d} / {:d} ... {:d}%".format(i+1, len(files), int((i+1) / len(files) * 100)))
            print("")
        self._motion_lengths = torch.tensor(motion_lengths, device=self.device, dtype=torch.long)
        
        print("===================================================")
        print("Loaded {:d} files with a total length of {:.3f}" \
            .format(self.num_motions, sum([length * self.dt for length in motion_lengths]))) # assert one motion_lib contain only one motion data
        print("===================================================")
        
    def reset(self, env_ids):
        self._motion_idxs[env_ids] = torch.randint(0, self.num_motions, (len(env_ids),), device=self.device)
        self._motion_offsets[env_ids] = torch.full((env_ids.shape[0],), self.min_past_frame, device=self.device, dtype=torch.long)
            
    def step_motion_state(self):
        self._motion_offsets += 1
        done = self._motion_offsets + self.min_future_frame >= self._motion_lengths[self._motion_idxs]
        
        return done
    
    def get_motion_state(self, time_offset=0.0):
        frame_offset = int(time_offset / self.dt)
        state = torch.zeros(self.num_envs, 13+2*self.num_dofs+10*self.num_bodies, device=self.device, dtype=torch.float32)
        for env_id in torch.arange(0, self.num_envs, device=self.device, dtype=torch.long):
            motion_idx = self._motion_idxs[env_id]
            state[env_id] = self._motion_state_tensors[motion_idx][self._motion_offsets[env_id] + frame_offset]
        
        return MotionState(state, self.num_dofs, self.num_bodies)