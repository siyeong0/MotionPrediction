import numpy as np
import torch

from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

class Motion:
    def __init__(self, files, num_envs, dt, start_limit, min_length, num_dof, num_bodies, device):
        self.num_envs = num_envs
        self.dt = dt
        self.start_limit = start_limit
        self.min_length = min_length
        # load motions
        self.motion_libs = []
        for file in files:
            motion_file = file if "." in file else f"{file}.npy" 
            motion_lib = MotionLib(motion_file=motion_file,
                                   num_dofs=num_dof,
                                   key_body_ids=np.array([x for x in range(num_bodies)]),
                                   device=device)
            self.motion_libs.append(motion_lib) 
        print("===================================================")
        print("Loaded {:d} motion files with a total length of {:.3f}" \
            .format(len(self.motion_libs), sum([motion.get_motion_length(0) for motion in self.motion_libs]))) # assert one motion_lib contain only one motion data
        print("===================================================")
            
        
        self.motion_idxs = np.full((num_envs), -1)
        self.motion_ids = np.full((num_envs), -1)
        self.curr_time = np.full((num_envs), -1.0, dtype=np.float32)
        
    def reset(self, env_ids):
        env_ids = env_ids.cpu().numpy()
        for env_id in env_ids:
            self.motion_idxs[env_id] = np.random.randint(len(self.motion_libs))  
            self.motion_ids[env_id] = self.sample_motion(env_id)
            #self.curr_time[env_id] = self.sample_time(env_id)
            self.curr_time[env_id] = 0.0
            
    def get_motion_lib(self, env_id) -> MotionLib:
        return self.motion_libs[self.motion_idxs[env_id]]
    
    def get_motion(self, env_id):
        return self.get_motion_lib(env_id).get_motion(self.motion_ids[env_id])
    
    def sample_motion(self, env_id):
        return self.get_motion_lib(env_id).sample_motions(1)[0]
    
    def get_motion_length(self, env_id):
        return self.get_motion_lib(env_id).get_motion_length(self.motion_ids[env_id])
    
    def sample_time(self, env_id):
        assert(self.get_motion_length(env_id) > (self.start_limit + self.min_length))
        while True:
            time = self.get_motion_lib(env_id).sample_time(np.array([self.motion_ids[env_id]]))[0]
            if time > self.start_limit and time < self.get_motion_length(env_id) - self.min_length:
                return time
            
    def get_motion_state(self, env_ids):
        env_ids = env_ids.cpu().numpy()
        
        states = []
        dones = []
        for env_id in env_ids:
            states.append(list(self.get_motion_lib(env_id).get_motion_state(
                np.array([self.motion_ids[env_id]]), np.array([self.curr_time[env_id]]))))
            self.curr_time[env_id] += self.dt
            dones.append(self.curr_time[env_id] > self.get_motion_length(env_id))
            
        root_pos = torch.stack([s[0][0] for s in states], dim=0)
        root_rot = torch.stack([s[1][0] for s in states], dim=0)
        dof_pos = torch.stack([s[2][0] for s in states], dim=0)
        root_vel = torch.stack([s[3][0] for s in states], dim=0)
        root_ang_vel = torch.stack([s[4][0] for s in states], dim=0)
        dof_vel = torch.stack([s[5][0] for s in states], dim=0)
        key_pos = torch.stack([s[6][0] for s in states], dim=0)
            
        return (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos), dones
            
        