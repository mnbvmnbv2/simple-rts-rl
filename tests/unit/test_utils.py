import jax.numpy as jnp

from src.rts.env import Board, EnvState
from src.rts.utils import get_legal_moves, fixed_argwhere


# Helper function to create an EnvState from lists.
def _create_test_state(p1, p2, neutral, bases, time=5):
    """
    Helper function to create an EnvState from lists.

    p1, p2, neutral: 2D lists (4x4) of ints representing the troops
                      for player 1, player 2 and the neutral troops.
    bases: 2D list (4x4) of booleans indicating base positions.

    The function stacks p1 and p2 to form the combined `player_troops` array.
    """
    board = Board(
        player_troops=jnp.stack(
            [jnp.array(p1, dtype=jnp.int32), jnp.array(p2, dtype=jnp.int32)], axis=0
        ),
        neutral_troops=jnp.array(neutral, dtype=jnp.int32),
        bases=jnp.array(bases, dtype=bool),
    )
    return EnvState(board=board, time=time)


def test_get_legal_moves_no_moves():
    """
    Test that if all cells have only 1 troop,
    no legal moves are available for player 1.
    """
    p1 = jnp.ones((4, 4), dtype=jnp.int32)
    p2 = jnp.zeros((4, 4), dtype=jnp.int32)
    neutral = jnp.zeros((4, 4), dtype=jnp.int32)
    bases = jnp.zeros((4, 4), dtype=bool)
    state = _create_test_state(p1, p2, neutral, bases)
    legal_moves = get_legal_moves(state, player=0)
    # Expect no cell to have a legal move (all False)
    assert jnp.all(legal_moves == False)


def test_fixed_argwhere_all_false():
    """
    Test that fixed_argwhere returns a count of 0 and all filler values
    when the mask is entirely False.
    """
    mask = jnp.zeros((4, 4, 4), dtype=bool)
    indices, num_actions = fixed_argwhere(mask, max_actions=10)
    assert num_actions == 0
    assert jnp.all(indices == -1)


def test_move_from_top_left_corner_valid():
    """
    Test moving from the top-left corner (0,0) where only right and down moves
    are legal. Verify that get_legal_moves produces the correct mask.
    """
    # Create a 2x2 board with an active cell at (0,0)
    p1 = [[5, 0], [0, 0]]
    p2 = [[0, 0], [0, 0]]
    neutral = [[0, 0], [0, 0]]
    bases = [[True, False], [False, False]]
    state = _create_test_state(p1, p2, neutral, bases)
    # Retrieve legal moves for player 1
    legal_moves = get_legal_moves(state, player=0)
    # For cell (0,0) with our [up, right, down, left] order,
    # expected: up: False, right: True, down: True, left: False.
    cell_moves = legal_moves[0, 0, :]
    expected = jnp.array([False, True, True, False])
    assert jnp.all(cell_moves == expected)


def test_multiple_legal_moves_fixed_argwhere():
    """
    Test that a cell with many legal moves returns all legal moves from fixed_argwhere.
    """
    # Create a 3x3 board with an active cell at (1,1) with 10 troops.
    p1 = jnp.zeros((3, 3), dtype=jnp.int32).at[1, 1].set(10)
    p2 = jnp.zeros((3, 3), dtype=jnp.int32)
    neutral = jnp.zeros((3, 3), dtype=jnp.int32)
    bases = jnp.zeros((3, 3), dtype=jnp.bool_)
    state = _create_test_state(p1, p2, neutral, bases)
    legal_moves = get_legal_moves(state, player=0)
    # For cell (1,1), all four directions should be legal.
    cell_moves = legal_moves[1, 1, :]
    assert jnp.all(cell_moves)
    indices, num_actions = fixed_argwhere(legal_moves, max_actions=10)
    # At least 4 legal moves should be found in the full board.
    assert num_actions >= 4


def test_get_legal_moves_boundaries():
    """
    Verify that cells on the board edges have the appropriate directional moves disabled.
    """
    # Top row: cell (0,1) should not allow an upward move.
    p1 = jnp.zeros((3, 3), dtype=jnp.int32).at[0, 1].set(3)
    p2 = jnp.zeros((3, 3), dtype=jnp.int32)
    neutral = jnp.zeros((3, 3), dtype=jnp.int32)
    bases = jnp.zeros((3, 3), dtype=bool)

    state = _create_test_state(p1, p2, neutral, bases)
    legal = get_legal_moves(state, player=0)

    cell = legal[0, 1]
    assert not cell[0]  # up
    assert cell[1]  # right
    assert cell[2]  # down
    assert cell[3]  # left

    # Bottom row: cell (2,1) should not allow a downward move.
    p1_n = jnp.zeros((3, 3), dtype=jnp.int32).at[2, 1].set(3)
    p2_n = jnp.zeros((3, 3), dtype=jnp.int32)
    neutral_n = jnp.zeros((3, 3), dtype=jnp.int32)
    bases_n = jnp.zeros((3, 3), dtype=jnp.bool_)
    state2 = _create_test_state(p1_n, p2_n, neutral_n, bases_n)
    legal_moves2 = get_legal_moves(state2, player=0)
    cell_moves2 = legal_moves2[2, 1]
    assert cell_moves2[0]  # Up
    assert cell_moves2[1]  # Right
    assert not cell_moves2[2]  # Down move not allowed on bottom row
    assert cell_moves2[3]  # Left
