import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import EnvState, increase_troops, init_state, move


@pytest.fixture
def params():
    return EnvConfig(
        board_width=10,
        board_height=10,
        num_neutral_bases=4,
        num_neutral_troops_start=8,
        neutral_troops_min=1,
        neutral_troops_max=10,
        player_start_troops=5,
    )


@pytest.fixture
def init_rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def state(init_rng, params):
    return init_state(init_rng, params)


def test_move_benchmark(benchmark, state: EnvState):
    benchmark(move, state, player=1, x=1, y=1, action=1)


def test_increase_troops_benchmark(benchmark, state: EnvState):
    benchmark(increase_troops, state)
