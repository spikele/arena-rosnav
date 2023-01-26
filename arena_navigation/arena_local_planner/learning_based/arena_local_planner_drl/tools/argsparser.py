import argparse
import os
import numpy as np

from tools.custom_mlp_utils import get_net_arch


def training_args(parser):
    """program arguments training script"""
    parser.add_argument(
        "--n_envs", type=int, default=1, help="number of parallel environments"
    )
    parser.add_argument(
        "--no-gpu", action="store_true", help="disables gpu for training"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="disables multiprocessing in order to debug",
    )
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--agent",
        type=str,
        #choices=AgentFactory.registry.keys(), # removed because it was causing a bug where the training script had to be restarted for actions_in_observationspace to be correct
        help="predefined agent to train",
    )
    group.add_argument(
        "--custom-mlp",
        action="store_true",
        help="enables training with custom multilayer perceptron",
    )
    group.add_argument(
        "--load",
        type=str,
        metavar="[agent name]",
        help="agent to be loaded for training",
    )
    parser.add_argument(
        "--config",
        type=str,
        metavar="[config name]",
        default="default",
        help="name of the json file containing" "the hyperparameters",
    )
    parser.add_argument(
        "--n", type=int, help="timesteps in total to be generated for training"
    )
    parser.add_argument(
        "-log",
        "--eval_log",
        action="store_true",
        help="enables storage of evaluation data",
    )
    parser.add_argument(
        "--tb", action="store_true", help="enables tensorboard logging"
    )
    parser.add_argument(
        "--info_file", action="store_true", help="enables saving additional information in a file"
    )


def off_policy_training_args(parser):
    """off-policy algorithm arguments"""
    parser.add_argument(
        "--load_replay_buffer", type=str, help="name of the agent whose replay buffer should be loaded"
    )
    parser.add_argument(
        "--save_replay_buffer", action="store_false", help="whether the replay buffer should be saved when the model is saved"
    )


def imitation_training_args(parser):
    """imitation learning arguments"""
    parser.add_argument(
        "--il_alg", type=str, default="", choices=["", "bc", "dagger"], help="IL algorithm to use"
    )
    parser.add_argument(
        "--recording", type=str, default="", help="file name of the recording"
    )
    parser.add_argument(
        "--expert", type=str, default="", help="name of the agent who will be used as expert for dagger"
    )
    parser.add_argument(
        "--eval_il", action="store_true", help="enables evaluating the IL-trained policy"
    )


def bc_training_args(parser):
    """behaviour cloning training script arguments"""
    parser.add_argument(
        "--recording", type=str, help="file name of the recording", required=True
    )


def dagger_training_args(parser):
    """dagger training script arguments"""
    parser.add_argument(
        "--expert", type=str, help="name of the agent who will be used as expert for dagger", required=True
    )


def run_agent_args(parser):
    parser.add_argument(
        "--no-gpu", action="store_true", help="disables gpu for training"
    )
    parser.add_argument(
        "--load",
        type=str,
        metavar="[agent name]",
        help="agent to be loaded for training",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="store log file with episode information",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        metavar="[scenario name]",
        default="scenario1",
        help="name of scenario file for deployment",
    )
    parser.add_argument(
        "--num_eps",
        type=int,
        metavar="[num episodes]",
        default=100,
        help="number of episodes the agent/s get/s challenged",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        metavar="[max steps per episode]",
        default=np.inf,
        help="max amount of actions per episode before the simulation is resetted",
    )
    parser.add_argument("-v", "--verbose", choices=["0", "1"], default="1")


def custom_mlp_args(parser):
    """arguments for the custom mlp mode"""
    custom_mlp_args = parser.add_argument_group(
        "custom mlp args", "architecture arguments for the custom mlp"
    )

    custom_mlp_args.add_argument(
        "--body",
        type=str,
        default="",
        metavar="{num}-{num}-...",
        help="architecture of the shared latent network, "
        "each number representing the number of neurons per layer",
    )
    custom_mlp_args.add_argument(
        "--pi",
        type=str,
        default="",
        metavar="{num}-{num}-...",
        help="architecture of the latent policy network, "
        "each number representing the number of neurons per layer",
    )
    custom_mlp_args.add_argument(
        "--vf",
        type=str,
        default="",
        metavar="{num}-{num}-...",
        help="architecture of the latent value network, "
        "each number representing the number of neurons per layer",
    )
    custom_mlp_args.add_argument(
        "--qf",
        type=str,
        default="",
        metavar="{num}-{num}-...",
        help="architecture of the latent q network, "
        "each number representing the number of neurons per layer",
    )
    custom_mlp_args.add_argument(
        "--act_fn",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh"],
        help="activation function to be applied after each hidden layer",
    )


def process_imitation_args(parsed_args):
    """argument check function for IL related arguments"""
    if parsed_args.il_alg == "bc":
        assert parsed_args.recording != ""
    if parsed_args.il_alg == "dagger":
        assert parsed_args.expert != ""
    

def process_training_args(parsed_args):
    """argument check function"""
    if parsed_args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if parsed_args.custom_mlp:
        setattr(parsed_args, "net_arch", get_net_arch(parsed_args))
    else:
        if (
            parsed_args.body != ""
            or parsed_args.pi != ""
            or parsed_args.vf != ""
            or parsed_args.qf != ""
        ):
            print("[custom mlp] arguments will be ignored..")
        delattr(parsed_args, "body")
        delattr(parsed_args, "pi")
        delattr(parsed_args, "vf")
        delattr(parsed_args, "qf")
        delattr(parsed_args, "act_fn")


def process_run_agent_args(parsed_args):
    if parsed_args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def parse_training_args(args=None, ignore_unknown=False):
    """parser for training script"""
    arg_populate_funcs = [training_args, off_policy_training_args, imitation_training_args, custom_mlp_args]
    arg_check_funcs = [process_training_args, process_imitation_args]

    return parse_various_args(
        args, arg_populate_funcs, arg_check_funcs, ignore_unknown
    )


def parse_off_policy_training_args(args=None, ignore_unknown=False):
    """parser for training script"""
    arg_populate_funcs = [training_args, custom_mlp_args, off_policy_training_args]
    arg_check_funcs = [process_training_args]

    return parse_various_args(
        args, arg_populate_funcs, arg_check_funcs, ignore_unknown
    )


def parse_bc_training_args(args=None, ignore_unknown=False):
    """parser for training script"""
    arg_populate_funcs = [training_args, custom_mlp_args, bc_training_args]
    arg_check_funcs = [process_training_args]

    return parse_various_args(
        args, arg_populate_funcs, arg_check_funcs, ignore_unknown
    )


def parse_dagger_training_args(args=None, ignore_unknown=False):
    """parser for training script"""
    arg_populate_funcs = [training_args, custom_mlp_args, bc_training_args, dagger_training_args]
    arg_check_funcs = [process_training_args]

    return parse_various_args(
        args, arg_populate_funcs, arg_check_funcs, ignore_unknown
    )


def parse_run_agent_args(args=None, ignore_unknown=False):
    """parser for training script"""
    arg_populate_funcs = [run_agent_args]
    arg_check_funcs = [process_run_agent_args]

    return parse_various_args(
        args, arg_populate_funcs, arg_check_funcs, ignore_unknown
    )


def parse_various_args(
    args, arg_populate_funcs, arg_check_funcs, ignore_unknown
):
    """generic arg parsing function"""
    parser = argparse.ArgumentParser()

    for func in arg_populate_funcs:
        func(parser)

    if ignore_unknown:
        parsed_args, unknown_args = parser.parse_known_args(args=args)
    else:
        parsed_args = parser.parse_args(args=args)
        unknown_args = []

    for func in arg_check_funcs:
        func(parsed_args)

    print_args(parsed_args)
    return parsed_args, unknown_args


def print_args(args):
    print("\n-------------------------------")
    print("            ARGUMENTS          ")
    for k in args.__dict__:
        print("- {} : {}".format(k, args.__dict__[k]))
    print("--------------------------------\n")
