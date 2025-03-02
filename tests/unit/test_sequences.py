import jax
import jax.numpy as jnp
import pytest

from src.rts.env import Board, EnvState, init_state, move, reinforce_troops
from src.rts.utils import assert_valid_state, get_legal_moves, fixed_argwhere
from src.rts.config import EnvConfig


def test_random_sequence_validity():
    """
    Run a sequence of moves (both players) on a small board to ensure the state remains valid.
    """
    config = EnvConfig(
        board_width=5,
        board_height=5,
        num_neutral_bases=2,
        num_neutral_troops_start=3,
        neutral_troops_min=2,
        neutral_troops_max=4,
        player_start_troops=5,
        bonus_time=10,
    )
    state = init_state(jax.random.PRNGKey(123), config)
    for _ in range(50):
        # Here we use dummy moves (which may be invalid); we simply verify that state remains valid.
        state = move(state, player=0, x=2, y=2, action=1)
        state = move(state, player=1, x=2, y=2, action=3)
        state = reinforce_troops(state, config)
        assert_valid_state(state)
