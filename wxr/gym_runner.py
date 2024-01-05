import threading

from isaacgym import gymapi

import torch

from mtss_sim.mtss_sim import MotionTrackingSim
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

from isaacgymenvs.utils.torch_jit_utils import quat_rotate, quat_mul

from wxr.common import *
from wxr.util import *

class GymRunner:
    def __init__(self, num_envs, model_path, sim_device='cuda'):
        self.sim_prev_idx = -1
        self.num_envs = num_envs
        self.env_cfg = MtssCfg()
        self.rl_cfg = MtssPPOCfg()
        self.model_path = model_path
        self.sim_device = sim_device
        self.sim = MotionTrackingSim(self.env_cfg, 
                                     self.num_envs, 
                                     self.rl_cfg, 
                                     self.model_path, 
                                     gymapi.SIM_PHYSX, 
                                     self.sim_device, False)
        self.motion_time_stride = self.env_cfg.env.time_stride

        NUM_PAST_FRAMES = self.env_cfg.env.num_past_frame
        NUM_FUTURE_FRAMES = self.env_cfg.env.num_future_frame
        BUF_SIZE = 1 + NUM_PAST_FRAMES + NUM_FUTURE_FRAMES
        
        self.head_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 7))
        self.left_hand_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 3))
        self.right_hand_input_buf = torch.zeros((BUF_SIZE, MAX_NUM_ENVS, 3))
        
        self.reset_buf = torch.full((MAX_NUM_ENVS,), fill_value=2, dtype=torch.long)
        self.reset_idx_buf = torch.zeros((MAX_NUM_ENVS), dtype=torch.long)
        self.head_offset = torch.zeros((MAX_NUM_ENVS), dtype=torch.float32)
        self.height_scale = torch.ones((MAX_NUM_ENVS), dtype=torch.float32)
        
        self.lock = threading.Lock()
        self._sim_prev_idx = -1
    
    def step(self, head_state_buf, left_hand_state_buf, right_hand_state_buf):
        sim = self.sim        
        head_input_buf = self.head_input_buf
        left_hand_input_buf = self.left_hand_input_buf
        right_hand_input_buf = self.right_hand_input_buf
        reset_buf = self.reset_buf
        reset_idx_buf = self.reset_idx_buf
        head_offset = self.head_offset
        height_scale = self.height_scale
        
        NUM_PAST_FRAMES = self.env_cfg.env.num_past_frame
        NUM_FUTURE_FRAMES = self.env_cfg.env.num_future_frame
        BUF_SIZE = 1 + NUM_PAST_FRAMES + NUM_FUTURE_FRAMES
        
        curr_time = round_to_sliced_time(time.time())
        time_offset = curr_time - get_init_time() - self.motion_time_stride * (self.env_cfg.env.num_future_frame + 1)
        idx = round(time_offset * SIM_FPS) % STATE_BUFFER_SIZE
        
        # syncronize with wxr
        if idx == self._sim_prev_idx:
            return None
        self._sim_prev_idx = idx
        
        # reset environments
        reset_env_ids = (reset_buf == 1).nonzero(as_tuple=False).flatten().to('cuda')
        if len(reset_env_ids) > 0:
            reset_root_state = torch.zeros((len(reset_env_ids), 13), dtype=torch.float32)
            for reset_env_id in reset_env_ids:
                reset_idx = reset_idx_buf[reset_env_id]
                reset_root_state[:, 0:7] = head_state_buf[reset_idx, reset_env_id]
                reset_root_pos = reset_root_state[:, 0:3]
                reset_root_quat = reset_root_state[:, 3:7]
                reset_root_pos[:, 2] = 0.90    # default root pos when dofs are zeroset
                reset_root_quat[:, 0:2] = 0.0  # extract yaw only
                reset_root_quat = torch.nn.functional.normalize(reset_root_quat, dim=1)
                
                head_state_buf[:, reset_env_id, :] = head_state_buf[reset_idx, reset_env_id]
                left_hand_state_buf[:, reset_env_id, :] = left_hand_state_buf[reset_idx, reset_env_id]
                right_hand_state_buf[:, reset_env_id, :] = right_hand_state_buf[reset_idx, reset_env_id]
            sim.reset_idx(reset_env_ids)
            sim.init_env_state(reset_env_ids, reset_root_state.to(SIM_DEVICE))
            reset_buf[reset_env_ids] = 0
        
        # step simulation
        base_idx = idx - NUM_PAST_FRAMES
        for fidx in range(BUF_SIZE):
            tidx = (base_idx + fidx) % STATE_BUFFER_SIZE
            head_state = torch.clone(head_state_buf[tidx, :, 0:7])
            head_pos = head_state[:, 0:3]
            head_quat = head_state[:, 3:7]
            lhp = left_hand_state_buf[tidx, :, 0:3]
            rhp = right_hand_state_buf[tidx, :, 0:3]
            scaled_lhp = lhp# * height_scale.unsqueeze(1).repeat(1, 3)
            scaled_rhp = rhp# * height_scale.unsqueeze(1).repeat(1, 3)
            glob_left_hand_pos = quat_rotate(head_quat, scaled_lhp) + head_pos
            glob_right_hand_pos = quat_rotate(head_quat, scaled_rhp) + head_pos
            
            head_input_buf[fidx, :, :] = head_state
            left_hand_input_buf[fidx, :, :] = glob_left_hand_pos
            right_hand_input_buf[fidx, :, :] = glob_right_hand_pos
        
        sim.step(head_input_buf,
                 left_hand_input_buf,
                 right_hand_input_buf)
        
        root_state_buf = sim.root_state.to('cpu')
        link_state_buf = sim.link_state.to('cpu')
        
        return root_state_buf, link_state_buf