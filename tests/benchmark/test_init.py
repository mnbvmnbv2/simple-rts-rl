import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import init_state


@pytest.fixture
def config():
    return EnvConfig(
        num_players=2,
        board_width=10,
        board_height=10,
        num_neutral_bases=4,
        num_neutral_troops_start=8,
        neutral_troops_min=1,
        neutral_troops_max=10,
        player_start_troops=5,
        bonus_time=10,
    )


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


def test_init_benchmark(benchmark, rng_key, config):
    benchmark(init_state, rng_key, config)
