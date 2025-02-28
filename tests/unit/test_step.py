import jax.numpy as jnp
import pytest
from src.rts.env import EnvState, increase_troops
from src.rts.utils import assert_valid_state


@pytest.fixture
def board():
    return jnp.array(
        [
            [
                [4, 0, 0, 0],  # 0, 0 p1
                [0, 0, 0, 0],
                [2, 0, 0, 1],  # 0, 2 p1 w base
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 0, 1],  # 1, 1 p2 w base
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 8, 0, 0],  # 2, 2 p2
                [0, 0, 3, 0],  # 2, 3 neutral
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 6, 1],  # 3, 2 neutral w base
                [0, 0, 0, 0],
            ],
        ]
    )


def test_increase_troops(board: jnp.array):
    state = EnvState(board=board, time=4)
    state = increase_troops(state)
    assert_valid_state(state)

    # check random two blank tiles
    assert jnp.all(state.board[0, 1, :] == 0)
    assert jnp.all(state.board[1, 2, :] == 0)

    # check that board is updated correctly
    # no bonus troops
    assert state.board[0, 0, 0] == 4
    assert state.board[0, 2, 0] == 3
    assert state.board[1, 1, 1] == 2
    assert state.board[2, 2, 1] == 8
    assert state.board[2, 3, 2] == 3
    assert state.board[3, 2, 2] == 6


def test_increase_troops_bonus(board: jnp.array):
    state = EnvState(board=board, time=0)
    state = increase_troops(state)
    assert_valid_state(state)

    # check random two blank tiles
    assert jnp.all(state.board[0, 1, :] == 0)
    assert jnp.all(state.board[1, 2, :] == 0)

    # check that board is updated correctly
    # with bonus troops
    assert state.board[0, 0, 0] == 5
    assert state.board[0, 2, 0] == 4
    assert state.board[1, 1, 1] == 3
    assert state.board[2, 2, 1] == 9
    assert state.board[2, 3, 2] == 3
    assert state.board[3, 2, 2] == 6
