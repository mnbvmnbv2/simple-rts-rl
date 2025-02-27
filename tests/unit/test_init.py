import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import init_state
from src.rts.utils import assert_valid_state


@pytest.fixture
def params():
    return EnvConfig(
        board_width=10,
        board_height=10,
        num_neutral_bases=4,
        num_neutral_troops_start=8,
        neutral_bases_min_troops=1,
        neutral_bases_max_troops=10,
    )


def test_init_state(params):
    for i in range(100):
        init_rng = jax.random.PRNGKey(i)
        state = init_state(init_rng, params)
        assert_valid_state(state)
