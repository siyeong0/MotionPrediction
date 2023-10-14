from mtss_gym.envs.mtss import MotionTrackingFromSparseSensor
from mtss_gym.envs.mtss_cfg import MtssCfg

from isaacgym import gymapi

if __name__ == "__main__":
    cfg = MtssCfg()
    env = MotionTrackingFromSparseSensor(cfg, gymapi.SIM_PHYSX, "cuda:0", False)
    obs = env.get_observations()
    
    while True:
        actions = env.sample_action()
        obs, _, rews, dones, infos = env.step(actions)  
      