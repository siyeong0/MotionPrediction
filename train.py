from isaacgym import gymapi

from mtss_gym.utils.helpers import class_to_dict, get_log_dir

from rsl_rl.runners import OnPolicyRunner

from mtss_gym.envs.mtss import MotionTrackingFromSparseSensor
from mtss_gym.envs.mtss_cfg import MtssCfg, MtssPPOCfg

if __name__ == "__main__":
    env = MotionTrackingFromSparseSensor(MtssCfg(), gymapi.SIM_PHYSX, "cuda:0", False)

    ppo_runner = OnPolicyRunner(env, class_to_dict(MtssPPOCfg), get_log_dir("logs", "mtss"), device="cuda:0")
    ppo_runner.learn(num_learning_iterations=MtssPPOCfg.runner.max_iterations, init_at_random_ep_len=True)