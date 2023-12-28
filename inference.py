import os

from isaacgym import gymapi

from rsl_rl.runners import OnPolicyRunner

from utils.helpers import class_to_dict, get_log_dir

from mtss_gym.mtss_gym import MotionTrackingFromSparseSensor
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

path = "model_17100.pt"

if __name__ == "__main__":
    # load environment
    env_cfg = MtssCfg()
    env_cfg.env.num_envs = 1
    env_cfg.motion.dir = "resources/motions"
    env_cfg.motion.files = os.listdir(env_cfg.motion.dir)
    env = MotionTrackingFromSparseSensor(env_cfg, gymapi.SIM_PHYSX, "cuda", False)
    # load model
    rl_cfg = MtssPPOCfg()
    rl_cfg.runner.resume = True
    ppo_runner = OnPolicyRunner(env, class_to_dict(MtssPPOCfg), get_log_dir("logs", "mtss"), device="cuda")
    ppo_runner.load(path)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # run    
    env.reset()
    obs = env.get_observations()
    while True:
        action = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(action.detach())
        