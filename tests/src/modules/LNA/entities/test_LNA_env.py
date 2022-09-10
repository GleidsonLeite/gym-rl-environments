from random import random
from src.modules.LNA.entities import LNA
from src.modules.LNA.utils.Normalization import clamp, z_normalize

import numpy as np


def test_it_should_be_able_to_initialize_the_state_with_random_values():
    env = LNA()
    observation = env.reset()
    assert all(
        feature <= env.MAX_RL or feature >= env.MIN_RL for feature in observation
    )


def test_it_should_be_able_to_take_positive_and_negative_actions():
    env = LNA()
    env.MAX_RL = random()
    env.MIN_RL = -random()
    observation = env.reset()
    action = [
        z_normalize(
            a=env.MIN_RL,
            b=env.MAX_RL,
            x=random(),
            x_max=1,
            x_min=0,
        )
        for _ in observation
    ]
    normalizer = 10
    expected_new_state = np.array(
        [
            clamp(
                max_=env.MAX_RL,
                min_=env.MIN_RL,
                value=feature
                + z_normalize(
                    a=-1,
                    b=1,
                    x=delta,
                    x_max=env.MAX_RL,
                    x_min=env.MIN_RL,
                )
                / normalizer,
            )
            for (delta, feature) in zip(action, observation)
        ],
        dtype=env.NDTYPE,
    )
    env._take_action(action=action, delta_normalizer=normalizer)
    new_observation = env._next_observation()

    assert np.allclose(expected_new_state, new_observation)
