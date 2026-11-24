from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1Decimation4Cfg( LeggedRobotCfg ):
    """G1 decimation 4: sim dt 0.005 s (200 Hz), control freq 50 Hz (policy dt 0.02 s)"""

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8]
        default_joint_angles = {
           'left_hip_yaw_joint' : 0.,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           'torso_joint' : 0.
        }

    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        episode_length_s = 30

    class commands( LeggedRobotCfg.commands ):
        class ranges:
            lin_vel_x = [0.5, 0.7]   # Moderate speed - works reliably
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.8, 0.8]
            heading = [-3.14, 3.14]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.0

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # Conservative PD gains
        stiffness = {'hip_yaw': 150,
                     'hip_roll': 150,
                     'hip_pitch': 150,
                     'knee': 200,
                     'ankle': 60,
                     }
        damping = {  'hip_yaw': 3.0,
                     'hip_roll': 3.0,
                     'hip_pitch': 3.0,
                     'knee': 6.0,
                     'ankle': 3.0,
                     }
        action_scale = 0.25
        decimation = 4  # control freq = 1/(decimation * sim.dt) = 50 Hz

    class sim( LeggedRobotCfg.sim ):
        dt = 0.005  # system frequency 200 Hz (same as default) - FIXED for all g1_decimation* configs

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78

        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.2
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.2
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.015
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1Decimation4CfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [256, 128, 64]  # MLP - lebih reliable
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCritic"  # MLP, bukan LSTM
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_decimation4'
