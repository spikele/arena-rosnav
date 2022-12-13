"""Custom policies built by SB3 during runtime through parsing 'policy_kwargs'"""
from torch import nn

from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.agent_factory_td3 import AgentFactoryTD3
from rl_agent.model.agent_factory_her import AgentFactoryHER
from rl_agent.model.agent_factory_sac import AgentFactorySAC
from rl_agent.model.base_agent import BaseAgent, PolicyType
from rl_agent.model.feature_extractors import *


@AgentFactory.register("AGENT_6")
class AGENT_6(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 64, 64, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_7")
class AGENT_7(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 128, 128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_8")
class AGENT_8(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [64, 64, 64, 64, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_9")
class AGENT_9(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [64, 64, 64, 64, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_10")
class AGENT_10(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [128, 128, 128, 128, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_11")
class AGENT_11(BaseAgent):
    type = PolicyType.MLP
    features_extractor_class = None
    features_extractor_kwargs = None
    net_arch = [512, 512, 512, 512, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_17")
class AGENT_17(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_4
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_18")
class AGENT_18(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_4
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_19")
class AGENT_19(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_20")
class AGENT_20(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[128], vf=[128])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_21")
class AGENT_21(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_22")
class AGENT_22(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=64)
    net_arch = [dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_23")
class AGENT_23(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_5
    features_extractor_kwargs = dict(features_dim=128)
    net_arch = [128, dict(pi=[64, 64, 64], vf=[64, 64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_frame_stacking")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6_FRAME_STACK
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_HER")
class AGENT_24(BaseAgent):
    type = PolicyType.MI
    features_extractor_class = EXTRACTOR_6_HER
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_frame_stacking_HER")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6_FRAME_STACK_HER
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU


"""@AgentFactory.register("AGENT_24_off_policy")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_frame_stacking_off_policy")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6_FRAME_STACK
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_HER_off_policy")
class AGENT_24(BaseAgent):
    type = PolicyType.MI
    features_extractor_class = EXTRACTOR_6_HER
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU


@AgentFactory.register("AGENT_24_frame_stacking_HER_off_policy")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6_FRAME_STACK_HER
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU"""


'''@AgentFactoryTD3.register("AGENT_24")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU

@AgentFactoryHER.register("AGENT_24")
class AGENT_24(BaseAgent):
    type = PolicyType.MI
    features_extractor_class = EXTRACTOR_6_HER
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU

@AgentFactorySAC.register("AGENT_24")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, 64, 64]
    activation_fn = nn.ReLU

@AgentFactory.register("AGENT_24_FrameStack")
class AGENT_24(BaseAgent):
    type = PolicyType.CNN
    features_extractor_class = EXTRACTOR_6_FrameStack
    features_extractor_kwargs = dict(features_dim=512)
    net_arch = [128, dict(pi=[64, 64], vf=[64, 64])]
    activation_fn = nn.ReLU'''
