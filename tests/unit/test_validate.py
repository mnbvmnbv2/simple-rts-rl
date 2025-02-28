import jax.numpy as jnp
import pytest

from src.rts.env import Board, EnvState
from src.rts.utils import assert_valid_state


@pytest.fixture
def board():
    return Board(
        player_1_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        player_2_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        neutral_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        bases=jnp.zeros((4, 4), dtype=jnp.bool_),
    )


def test_blank_state(board: Board):
    state = EnvState(board=board, time=0)
    assert_valid_state(state)


def test_basic_board(board: Board):
    board = board.player_1_troops.at[0, 0].set(1)
    state = EnvState(board=board, time=0)
    assert_valid_state(state)


def test_negative_time(board: Board):
    state = EnvState(board=board, time=-1)
    with pytest.raises(AssertionError):
        assert_valid_state(state)


def test_negative_troops(board: Board):
    board.player_1_troops = board.player_1_troops.at[0, 0].set(-1)
    state = EnvState(board=board, time=0)
    with pytest.raises(AssertionError):
        assert_valid_state(state)
