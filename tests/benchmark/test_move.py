import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import EnvState, init_state, move


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


@pytest.fixture
def init_rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def state(init_rng, params):
    return init_state(init_rng, params)


def test_move_benchmark(benchmark, state: EnvState):
    benchmark(move, state, player=1, x=1, y=1, action=1)
