import torch
from torch import nn
from torchrl.envs import GymEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torch import optim
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


class PPOAgent:
    def __init__(
            self,
            env:GymEnv,
            num_cells:int,
            frames_per_batch:int,
            total_frames:int,
            clip_epsilon,
            entropy_eps,
            gamma:float,
            lmbda:float,
            learning_rate:float,
            num_epochs:int,
            sub_batch_size:int,
            max_grad_norm:float,
            device:str
    ):
        self.device = device
        self.num_cells = num_cells
        self.env = env
        self.clip_epsilon = clip_epsilon
        self.entropy_eps = entropy_eps
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = learning_rate
        self.num_epochs = num_epochs
        self.sub_batch_size = sub_batch_size
        self.max_grad_norm = max_grad_norm


        self.policy_module = self._policyModule()
        self.value_module = self._valueModule()
        self.data_collector = self._dataCollectr()
        self.replay_buffer = self._replayBuffer()
        self.advantage_module = self._advantageModule()
        self.loss_module = self._ppoLoss()
        self.optimizer = self._optimizer()
        self.scheduler = self._scheduler()


    def _policyModule(self):
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


    def _valueModule(self):
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
    

    def _dataCollectr(self):
        collector = SyncDataCollector(
            create_env_fn=self.env,
            policy=self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device
        )
        return collector


    def _replayBuffer(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.frames_per_batch,
                device=self.device
            )
        )
        return rb


    def _advantageModule(self):
        advantage_module = GAE(
            gamma=self.gamma, 
            lmbda=self.lmbda, 
            value_network=self.value_module,
            average_gae=True
        )
        return advantage_module


    def _ppoLoss(self):
        loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=bool(self.entropy_eps),
            entropy_coef=self.entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )
        return loss_module


    def _optimizer(self):
        optimizer = optim.Adam(
            params=self.loss_module.parameters(),
            lr=self.lr
        )
        return optimizer
    

    def _scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            self.total_frames // self.frames_per_batch,
            0.0
        )
        return scheduler


    def train(self):
        self.logs = defaultdict(list)
        pbar = tqdm(total=self.total_frames)
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(self.data_collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(self.num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                self.advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replay_buffer.extend(data_view.cpu())
                for _ in range(self.frames_per_batch // self.sub_batch_size):
                    subdata = self.replay_buffer.sample(self.sub_batch_size)
                    loss_vals = self.loss_module(subdata.to(self.device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={self.logs['reward'][-1]: 4.4f} (init={self.logs['reward'][0]: 4.4f})"
            )
            self.logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {self.logs['step_count'][-1]}"
            self.logs["lr"].append(self.optimizer.param_groups[0]["lr"])
            lr_str = f"lr policy: {self.logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (1000, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = self.env.rollout(1000, self.policy_module)
                    self.logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    self.logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    self.logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                        f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                        f"eval step-count: {self.logs['eval step_count'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            self.scheduler.step()


    def plot_results(self):
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(self.logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(self.logs["eval reward (sum)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(self.logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()