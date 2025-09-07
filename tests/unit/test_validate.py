import jax.numpy as jnp
import pytest

from src.rts.env import Board, EnvState
from src.rts.utils import assert_valid_state


@pytest.fixture
def board():
    return Board(
        player_troops=jnp.zeros((2, 4, 4), dtype=jnp.int32),
        neutral_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        bases=jnp.zeros((4, 4), dtype=jnp.bool_),
    )


def test_blank_state(board: Board):
    state = EnvState(board=board, time=jnp.array(0, dtype=jnp.int32))
    assert_valid_state(state)


def test_basic_board(board: Board):
    # because frozen
    new_player_troops = board.player_troops.at[0, 0, 0].set(1)
    board = board.replace(player_troops=new_player_troops)
    state = EnvState(board=board, time=jnp.array(0, dtype=jnp.int32))
    assert_valid_state(state)


def test_negative_time(board: Board):
    state = EnvState(board=board, time=jnp.array(-1, dtype=jnp.int32))
    with pytest.raises(AssertionError):
        assert_valid_state(state)


def test_negative_troops(board: Board):
    # because frozen
    new_player_troops = board.player_troops.at[0, 0, 0].set(-1)
    board = board.replace(player_troops=new_player_troops)
    state = EnvState(board=board, time=jnp.array(0, dtype=jnp.int32))
    with pytest.raises(AssertionError):
        assert_valid_state(state)
