from isaacgym import gymapi

import torch

from mtss_sim.mtss_sim import MotionTrackingSim
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

from utils.motion import Motion
from isaacgymenvs.utils.torch_jit_utils import to_torch

if __name__ == "__main__":
    env_cfg = MtssCfg()
    rl_cfg = MtssPPOCfg()
    
    mt_sim = MotionTrackingSim(MtssCfg(), 32, MtssPPOCfg, "model_12700.pt", gymapi.SIM_PHYSX, "cuda", False)

    dir = env_cfg.motion.dir
    files = env_cfg.motion.files
    motion = Motion([f"{dir}/{file}" for file in files],
                            32, 
                            env_cfg.sim.dt,
                            env_cfg.env.num_past_frame * env_cfg.env.time_stride,
                            env_cfg.env.num_future_frame * env_cfg.env.time_stride,
                            28, 15,
                            'cuda')
    pos_sensor_indices = mt_sim.pos_sensor_indices
    hmd_index = pos_sensor_indices[0]
    left_hand_index = pos_sensor_indices[1]
    right_hand_index = pos_sensor_indices[2]
    
    mt_sim.init_env_state(to_torch([x for x in range(32)], dtype=torch.long, device='cuda'), motion.get_motion_state().root_state)
    
    while True:
        hmd_buf = []
        left_hand_buf = []
        right_hand_buf = []
        
        curr_motion_state = motion.get_motion_state(0.0)
        hmd_buf.append(curr_motion_state.link_state[:, hmd_index, 0:7])
        left_hand_buf.append(curr_motion_state.link_pos[:, left_hand_index])
        right_hand_buf.append(curr_motion_state.link_pos[:, right_hand_index])
        
        for i in range(env_cfg.env.num_past_frame):
            past_motion_state = motion.get_motion_state(-(i+1) * env_cfg.env.time_stride)
            hmd_buf.insert(0, past_motion_state.link_state[:, hmd_index, 0:7])
            left_hand_buf.insert(0, past_motion_state.link_pos[:, left_hand_index])
            right_hand_buf.insert(0, past_motion_state.link_pos[:, right_hand_index])
            
        for i in range(env_cfg.env.num_future_frame):
            future_motion_state = motion.get_motion_state((i+1) * env_cfg.env.time_stride)
            hmd_buf.append(future_motion_state.link_state[:, hmd_index, 0:7])
            left_hand_buf.append(future_motion_state.link_pos[:, left_hand_index])
            right_hand_buf.append(future_motion_state.link_pos[:, right_hand_index])
        
        done = motion.step_motion_state()
        env_ids = done.nonzero(as_tuple=False).flatten()
        mt_sim.reset_idx(env_ids)
        motion.reset(env_ids)
        mt_sim.init_env_state(env_ids, motion.get_motion_state().root_state)
        
        mt_sim.step(
            torch.stack(hmd_buf, dim=0),
            torch.stack(left_hand_buf, dim=0),
            torch.stack(right_hand_buf, dim=0))
        
        