from stable_baselines3 import SAC
from torch.nn.modules.activation import Tanh
from stable_baselines3.common.env_util import make_vec_env


from src.modules.LNA.entities import LNA

env = make_vec_env(LNA, n_envs=4, seed=0)


model = SAC(
    env=env,
    policy="MlpPolicy",
    verbose=1,
    policy_kwargs={"activation_fn": Tanh},
    batch_size=64,
    gradient_steps=2,
)
model.learn(total_timesteps=20000, log_interval=4)
