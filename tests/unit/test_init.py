import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import init_state
from src.rts.utils import assert_valid_state


# check for many EnvConfigs
@pytest.mark.parametrize(
    "params",
    [
        EnvConfig(
            board_width=10,
            board_height=10,
            num_neutral_bases=4,
            num_neutral_troops_start=8,
            neutral_bases_min_troops=1,
            neutral_bases_max_troops=10,
        ),
        EnvConfig(
            board_width=20,
            board_height=20,
            num_neutral_bases=4,
            num_neutral_troops_start=8,
            neutral_bases_min_troops=1,
            neutral_bases_max_troops=10,
        ),
        EnvConfig(
            board_width=12,
            board_height=47,
            num_neutral_bases=9,
            num_neutral_troops_start=3,
            neutral_bases_min_troops=8,
            neutral_bases_max_troops=11,
        ),
        EnvConfig(
            board_width=5,
            board_height=5,
            num_neutral_bases=2,
            num_neutral_troops_start=5,
            neutral_bases_min_troops=2,
            neutral_bases_max_troops=7,
        ),
    ],
)
def test_init_state(params):
    for i in range(100):
        init_rng = jax.random.PRNGKey(i)
        state = init_state(init_rng, params)
        assert_valid_state(params, state)
