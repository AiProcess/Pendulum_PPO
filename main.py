#%%
import torch
from Environment import EnvSetup, renderEnv
from Agent import PPOAgent
from torchrl.envs import check_env_specs
from torchrl.envs.utils import RandomPolicy

#%% Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Selected device is {device}')

#%% Hyperparameters
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0

frame_skips = 0
frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

#%% Environment
env_setup = EnvSetup(env_name="InvertedDoublePendulum-v4", device=device)
env = env_setup.create_main_env()
test_env = env_setup.create_test_env()

check_env_specs(env)
check_env_specs(test_env)

renderEnv(env=test_env, max_steps=1000, policy=RandomPolicy(test_env.action_spec))

#%% Agent
agent = PPOAgent(
    device=device, num_cells=num_cells, env=env,
    frames_per_batch=frames_per_batch, total_frames=total_frames, 
    clip_epsilon=clip_epsilon, entropy_eps=entropy_eps, gamma=gamma,
    lmbda=lmbda, learning_rate=lr, num_epochs=num_epochs,
    sub_batch_size=sub_batch_size, max_grad_norm=max_grad_norm
)

print("Running policy:", agent.policy_module(env.reset()))
print("Running value:", agent.value_module(env.reset()))

agent.train()
agent.plot_results()
renderEnv(env=test_env, max_steps=1000, policy=agent.policy_module)
# %%
