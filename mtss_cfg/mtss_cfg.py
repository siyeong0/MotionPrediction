import os
from .base_cfg import BaseConfig

class MtssCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 431
        num_past_frame = 3
        num_future_frame = 3
        time_stride = 1.0 / 3.0
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 28
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        max_episode_length_s = 180 # max episode length in seconds

    class asset:
        file = "resources/humanoid.xml"
        name = "humanoid"  # actor name
        disable_gravity = False
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 985
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.1
        thickness = 0.01
        
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        
        pos_sensor_body_names = ["head", "left_hand", "right_hand"]
        force_sensor_body_names = ["left_foot", "right_foot"]
        
    class motion:
        dir = "resources/test"
        files = os.listdir(dir)
        
    class reward:
        functions = ["imitation", "contact", "regularization"]
        # weights will be normalized
        class coef:
            w_i = 0.8
            w_c = 0.0
            w_r = 0.2
            class imitation:
                w_q = 0.55
                w_qv = 0.05
                w_p = 0.4
                w_pv = 0.05
                w_r = 0.1
                
                k_q = 40.0
                k_qv = 0.3
                k_p = 6.0
                k_pv = 2.0
                k_r = 3.0
            class contact:
                w_c = 1.0
                
                k_c = 0.5
            class regularization:
                w_a = 0.4
                w_s = 0.6
                
                k_a = 0.1
                k_s = 0.05
            
    class normalization:
        clip_observations = 100.
        clip_actions = 100.

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [4, 0, 2]  # [m]
        lookat = [0, 0, 1]  # [m]

    class sim:
        use_gpu = True
        dt = 1/36
        substeps = 2
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        
        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5

class MtssPPOCfg(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [300, 200, 100]
        critic_hidden_dims = [400, 400, 300, 200]
        activation = 'tanh' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 8 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.2e-4
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.97
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 15 # per iteration
        max_iterations = 100000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt