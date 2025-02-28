import jax.numpy as jnp
import pytest
from src.rts.env import Board, EnvState, increase_troops
from src.rts.utils import assert_valid_state


@pytest.fixture
def board():
    board = Board(
        player_1_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        player_2_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        neutral_troops=jnp.zeros((4, 4), dtype=jnp.int32),
        bases=jnp.zeros((4, 4), dtype=jnp.bool_),
    )
    # p1 troops at 0,0 and 0,2
    board.player_1_troops = board.player_1_troops.at[0, 0, 0].set(4)
    board.player_1_troops = board.player_1_troops.at[0, 2, 0].set(2)
    # p2 troops at 1,1 and 2,2
    board.player_2_troops = board.player_2_troops.at[1, 1, 1].set(1)
    board.player_2_troops = board.player_2_troops.at[2, 2, 1].set(8)
    # neutral troops at 2,3 and 3,2
    board.neutral_troops = board.neutral_troops.at[2, 3].set(3)
    board.neutral_troops = board.neutral_troops.at[3, 2].set(6)
    # bases at 0,2 1,1 3,2
    board.bases = board.bases.at[0, 2].set(True)
    board.bases = board.bases.at[1, 1].set(True)
    board.bases = board.bases.at[3, 2].set(True)

    print(board)

    return board


def test_increase_troops(board: jnp.array):
    state = EnvState(board=board, time=4)
    state = increase_troops(state)
    assert_valid_state(state)

    # check random two blank tiles
    assert state.board.player_1_troops[0, 1] == 0
    assert state.board.player_2_troops[0, 1] == 0
    assert state.board.neutral_troops[0, 1] == 0
    assert state.board.bases[0, 1] is False

    assert state.board.player_1_troops[1, 2] == 0
    assert state.board.player_2_troops[1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    assert state.board.bases[1, 2] is False

    # check that board is updated correctly
    # no bonus troops
    assert state.board.player_1_troops[0, 0] == 4
    assert state.board.player_1_troops[0, 2] == 3
    assert state.board.player_2_troops[1, 1] == 2
    assert state.board.player_2_troops[2, 2] == 8
    assert state.board.neutral_troops[2, 3] == 3
    assert state.board.neutral_troops[3, 2] == 6


def test_increase_troops_bonus(board: jnp.array):
    state = EnvState(board=board, time=0)
    state = increase_troops(state)
    assert_valid_state(state)

    # check random two blank tiles
    assert state.board.player_1_troops[0, 1] == 0
    assert state.board.player_2_troops[0, 1] == 0
    assert state.board.neutral_troops[0, 1] == 0
    assert state.board.bases[0, 1] is False

    assert state.board.player_1_troops[1, 2] == 0
    assert state.board.player_2_troops[1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    assert state.board.bases[1, 2] is False

    # check that board is updated correctly
    # with bonus troops
    assert state.board.player_1_troops[0, 0] == 5
    assert state.board.player_1_troops[0, 2] == 4
    assert state.board.player_2_troops[1, 1] == 3
    assert state.board.player_2_troops[2, 2] == 9
    assert state.board.neutral_troops[2, 3] == 3
    assert state.board.neutral_troops[3, 2] == 6
