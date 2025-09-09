import jax
import jax.numpy as jnp
import pytest

from src.rts.config import EnvConfig
from src.rts.env import init_state
from tests.helpers import assert_valid_state


# check for many EnvConfigs
@pytest.mark.parametrize(
    "config",
    [
        EnvConfig(
            num_players=2,
            board_width=10,
            board_height=10,
            num_neutral_bases=4,
            num_neutral_troops_start=8,
            neutral_troops_min=1,
            neutral_troops_max=10,
            player_start_troops=5,
            bonus_time=10,
        ),
        EnvConfig(
            num_players=3,
            board_width=20,
            board_height=20,
            num_neutral_bases=4,
            num_neutral_troops_start=8,
            neutral_troops_min=1,
            neutral_troops_max=10,
            player_start_troops=5,
            bonus_time=10,
        ),
        EnvConfig(
            num_players=4,
            board_width=12,
            board_height=47,
            num_neutral_bases=9,
            num_neutral_troops_start=3,
            neutral_troops_min=8,
            neutral_troops_max=11,
            player_start_troops=7,
            bonus_time=10,
        ),
        EnvConfig(
            num_players=2,
            board_width=5,
            board_height=5,
            num_neutral_bases=2,
            num_neutral_troops_start=5,
            neutral_troops_min=2,
            neutral_troops_max=7,
            player_start_troops=1,
            bonus_time=10,
        ),
    ],
)
def test_init_state(config):
    for i in range(100):
        init_rng = jax.random.PRNGKey(i)
        state = init_state(init_rng, config)
        assert_valid_state(state)


def test_init_state_minimal_board():
    """
    Test initialization on a minimal board where total_cells exactly equals num_special.
    For a 3x3 board and parameters chosen so that 2 (players) + num_neutral_bases + num_neutral_troops_start equals 9.
    """
    config = EnvConfig(
        num_players=2,
        board_width=3,
        board_height=3,
        num_neutral_bases=2,
        num_neutral_troops_start=5,  # 2 + 2 + 5 = 9
        neutral_troops_min=2,
        neutral_troops_max=5,
        player_start_troops=5,
        bonus_time=10,
    )
    state = init_state(jax.random.PRNGKey(42), config)
    # There should be exactly 2 (player bases) + 2 (neutral bases) marked as bases.
    num_bases = int(jnp.sum(state.board.bases))
    assert num_bases == 4
