from src.modules.LNA.entities import LNA

from stable_baselines3 import SAC
from torch.nn.modules.activation import Tanh

env = LNA()


model = SAC(
    env=env,
    policy="MlpPolicy",
    verbose=1,
    policy_kwargs={"activation_fn": Tanh},
)
model.learn(total_timesteps=20000)
