from mtss_gym.envs.mtss import MotionTrackingFromSparseSensor
from mtss_gym.envs.mtss_cfg import MtssCfg

from isaacgym import gymapi

if __name__ == "__main__":
    cfg = MtssCfg()
    env = MotionTrackingFromSparseSensor(cfg, gymapi.SIM_PHYSX, "cuda:0", False)
    env.reset()
    
    env.play()