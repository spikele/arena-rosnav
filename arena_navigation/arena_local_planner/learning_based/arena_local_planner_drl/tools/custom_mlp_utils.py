import torch as th
import argparse
import warnings

from tools.constants import OFF_POLICY_ALGORITHMS, ON_POLICY_ALGORITHMS

def get_net_arch(args: argparse.Namespace):
    """function to convert input args into valid syntax for the PPO"""
    body, policy, value = None, None, None

    if args.body != "":
        body = parse_string(args.body)
    if args.pi != "":
        policy = parse_string(args.pi)
    if args.vf != "":
        value = parse_string(args.vf)

    if body is None:
        body = []
    vf_pi = {}
    if value is not None:
        vf_pi["vf"] = value
    if policy is not None:
        vf_pi["pi"] = policy

    return body + [vf_pi]


def get_net_arch_thesis(args: argparse.Namespace):
    """function to convert input args into valid syntax for the PPO"""
    body, policy, value, q = None, None, None, None
    if args.rl_arg in ON_POLICY_ALGORITHMS:
        if args.body != "":
            body = parse_string(args.body)
        if args.pi != "":
            policy = parse_string(args.pi)
        if args.vf != "":
            value = parse_string(args.vf)
        if args.qf != "":
            warnings.warn("cannot use body for on-policy algorithms")

        if body is None:
            body = []
        vf_pi = {}
        if value is not None:
            vf_pi["vf"] = value
        if policy is not None:
            vf_pi["pi"] = policy

        return body + [vf_pi]
    
    elif args.rl_arg in OFF_POLICY_ALGORITHMS:
        if args.body != "":
            warnings.warn("cannot use body for off-policy algorithms")
        if args.pi != "":
            policy = parse_string(args.pi)
        if args.vf != "":
            warnings.warn("cannot use vf for off-policy algorithms")
        if args.qf != "":
            q = parse_string(args.qf)

        if q is None or policy is None:
            raise AssertionError("qf and pi are needed for off-policy algorithms")

        return {"qf": q, "pi": policy}


def parse_string(string: str):
    """function to convert a string into a int list

    Example:

    Input: parse_string("64-64")
    Output: [64, 64]

    """
    string_arr = string.split("-")
    int_list = []
    for string in string_arr:
        try:
            int_list.append(int(string))
        except:
            raise Exception("Invalid argument format on: " + string)
    return int_list


def get_act_fn(act_fn_string: str):
    """function to convert str into pytorch activation function class"""
    if act_fn_string == "relu":
        return th.nn.ReLU
    elif act_fn_string == "sigmoid":
        return th.nn.Sigmoid
    elif act_fn_string == "tanh":
        return th.nn.Tanh
