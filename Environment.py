from torchrl.envs import GymEnv
from torchrl.envs.transforms import (
    StepCounter, TransformedEnv, ObservationNorm, DoubleToFloat, Compose
)


class EnvSetup:
    def __init__(self, env_name:str, device:str):
        self.env_name = env_name
        self.device = device
        self.base_env = GymEnv(self.env_name, device=self.device)


    def create_main_env(self)->GymEnv:
        env = TransformedEnv(
            env=self.base_env,
            transform=Compose(
                ObservationNorm(in_keys=['observation']),
                DoubleToFloat(),
                StepCounter()
            )
        )
        env.transform[0].init_stats(1000)
        return env


    def create_test_env(self)->GymEnv:
        pass