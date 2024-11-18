from torchrl.envs import GymEnv
from torchrl.envs.transforms import (
    StepCounter, TransformedEnv, ObservationNorm, DoubleToFloat, Compose
)
from torchrl.record import CSVLogger, VideoRecorder
from torchrl.envs.utils import TensorDictBase


csv_logger = CSVLogger(exp_name="test", log_dir="log", video_format="mp4")
video_recorder = VideoRecorder(logger=csv_logger, tag="video")


class EnvSetup:
    def __init__(self, env_name:str, device:str):
        self.env_name = env_name
        self.device = device


    def create_main_env(self)->GymEnv:
        base_env = GymEnv(
            env_name=self.env_name, device=self.device
        )
        env = TransformedEnv(
            env=base_env,
            transform=Compose(
                ObservationNorm(in_keys=['observation']),
                DoubleToFloat(),
                StepCounter()
            )
        )
        env.transform[0].init_stats(1000)
        return env


    def create_test_env(self)->GymEnv:
        base_env = GymEnv(
            env_name=self.env_name,
            from_pixels=True,
            pixels_only=False,
            device=self.device
        )
        env = TransformedEnv(
            env=base_env,
            transform=video_recorder
        )
        return env
    

def renderEnv(env:GymEnv, max_steps:int, policy:TensorDictBase):
    env.rollout(max_steps=max_steps, policy=policy)
    video_recorder.dump()

