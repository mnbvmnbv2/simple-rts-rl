import jax.numpy as jnp
import pytest

from src.rts.env import EnvState
from src.rts.utils import assert_valid_state


def test_blank_state():
    state = EnvState(board=jnp.zeros((10, 10, 4), dtype=jnp.int32), time=0)
    assert_valid_state(state)


def test_basic_board():
    board = jnp.zeros((10, 10, 4), dtype=jnp.int32)
    board = board.at[0, 0, 0].set(1)
    state = EnvState(board=board, time=0)
    assert_valid_state(state)


def test_negative_time():
    board = jnp.zeros((10, 10, 4), dtype=jnp.int32)
    state = EnvState(board=board, time=-1)
    with pytest.raises(AssertionError):
        assert_valid_state(state)


def test_negative_troops():
    board = jnp.zeros((10, 10, 4), dtype=jnp.int32)
    board = board.at[0, 0, 0].set(-1)
    state = EnvState(board=board, time=0)
    with pytest.raises(AssertionError):
        assert_valid_state(state)
