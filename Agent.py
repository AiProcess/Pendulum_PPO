from torch import nn
from torchrl.envs import GymEnv
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator


class Agent:
    def __init__(self, device:str, num_cells:int, env:GymEnv):
        self.device = device
        self.num_cells = num_cells
        self.env = env


    def actor_net(self):
        net = nn.Sequential(
            nn.LazyLinear(out_features=self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(out_features=self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(out_features=self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(
                out_features=2*self.env.action_spec.shape[-1], device=self.device
            ),
            NormalParamExtractor()
        )

        td_net = TensorDictModule(
            net, in_keys=['observation'], out_keys=['loc', 'scale']
        )

        policy = ProbabilisticActor(
            module=td_net,
            spec=self.env.action_spec,
            in_keys=['loc', 'scale'],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low":self.env.action_spec.space.low,
                "high":self.env.action_spec.space.high
            },
            return_log_prob=True
        )
        return policy


    def value_net(self):
        value_net = nn.Sequential(
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(self.num_cells, device=self.device),
            nn.Tanh(),
            nn.LazyLinear(1, device=self.device),
        )

        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )

        return value_module
