from isaacgym import gymapi

from rsl_rl.runners import OnPolicyRunner

from mtss_gym.utils.helpers import class_to_dict, get_log_dir

from mtss_gym.envs.mtss import MotionTrackingFromSparseSensor
from mtss_gym.envs.mtss_cfg import MtssCfg, MtssPPOCfg

path = "model_1200.pt"

if __name__ == "__main__":
    # load environment
    env_cfg = MtssCfg()
    env_cfg.env.num_envs = 2
    env = MotionTrackingFromSparseSensor(env_cfg, gymapi.SIM_PHYSX, "cuda:0", False)
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
        