import jax
import pytest
from src.rts.config import EnvConfig
from src.rts.env import init_state


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
def rng_key():
    return jax.random.PRNGKey(0)


def test_init_benchmark(benchmark, rng_key, params):
    benchmark(init_state, rng_key, params)
