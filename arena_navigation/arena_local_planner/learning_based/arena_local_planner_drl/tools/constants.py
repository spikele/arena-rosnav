ON_POLICY_ALGORITHMS = ["ppo"]
OFF_POLICY_ALGORITHMS = ["sac", "tqc"]


HYPERPARAM_KEYS_GENERAL = {
    key: None
    for key in [
        "agent_name",
        "robot",
        "actions_in_observationspace",
        "reward_fnc",
        "discrete_action_space",
        "normalize",
        "task_mode",
        "train_max_steps_per_episode",
        "eval_max_steps_per_episode",
        "goal_radius",
        "curr_stage",
        "batch_size",
        #"use_frame_stacking",
        "seed",
        "algorithm",
        "use_frame_stacking",
    ]
}

HYPERPARAM_KEYS_PPO = {**HYPERPARAM_KEYS_GENERAL, **{
    key: None
    for key in [
        "gamma",
        "n_steps",
        "ent_coef",
        "learning_rate",
        "vf_coef",
        "max_grad_norm",
        "gae_lambda",
        "m_batch_size",
        "n_epochs",
        "clip_range",
    ]
}}

HYPERPARAM_KEYS_SAC = {**HYPERPARAM_KEYS_GENERAL, **{
    key: None
    for key in [
        "learning_rate",
        "buffer_size",
        "learning_starts",
        "m_batch_size",
        "tau",
        "gamma",
        #"n_steps",
        "train_freq",
        "gradient_steps",
        "action_noise",
        #"replay_buffer_class",
        #"replay_buffer_kwargs",
        "optimize_memory_usage",
        "ent_coef",
        "target_update_interval",
        "target_entropy",
        "use_sde",
        "sde_sample_freq",
        "use_sde_at_warmup",
        "use_her"
    ]
}}

HYPERPARAM_KEYS_TQC = {**HYPERPARAM_KEYS_SAC, **{"top_quantiles_to_drop_per_net": None}}