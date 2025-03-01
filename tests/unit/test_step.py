import chex
import jax.numpy as jnp
import pytest
from src.rts.env import Board, EnvState, increase_troops, move
from src.rts.utils import assert_valid_state


@pytest.fixture
def board():
    player_1_troops = jnp.zeros((4, 4), dtype=jnp.int32)
    player_2_troops = jnp.zeros((4, 4), dtype=jnp.int32)
    neutral_troops = jnp.zeros((4, 4), dtype=jnp.int32)
    bases = jnp.zeros((4, 4), dtype=jnp.bool_)
    # p1 troops at 0,0 and 0,2
    player_1_troops = player_1_troops.at[0, 0].set(4)
    player_1_troops = player_1_troops.at[0, 2].set(2)
    # p2 troops at 1,1 and 2,2
    player_2_troops = player_2_troops.at[1, 1].set(1)
    player_2_troops = player_2_troops.at[2, 2].set(8)
    # neutral troops at 2,3 and 3,2
    neutral_troops = neutral_troops.at[2, 3].set(3)
    neutral_troops = neutral_troops.at[3, 2].set(6)
    # bases at 0,2 1,1 3,2
    bases = bases.at[0, 2].set(True)
    bases = bases.at[1, 1].set(True)
    bases = bases.at[3, 2].set(True)

    board = Board(
        player_1_troops=player_1_troops,
        player_2_troops=player_2_troops,
        neutral_troops=neutral_troops,
        bases=bases,
    )

    return board


def test_increase_troops(board: jnp.array):
    state = EnvState(board=board, time=4)
    state = increase_troops(state)
    assert_valid_state(state)

    # check random two blank tiles
    assert state.board.player_1_troops[0, 1] == 0
    assert state.board.player_2_troops[0, 1] == 0
    assert state.board.neutral_troops[0, 1] == 0
    chex.assert_equal(state.board.bases[0, 1], False)

    assert state.board.player_1_troops[1, 2] == 0
    assert state.board.player_2_troops[1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    chex.assert_equal(state.board.bases[1, 2], False)

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
    chex.assert_equal(state.board.bases[0, 1], False)

    assert state.board.player_1_troops[1, 2] == 0
    assert state.board.player_2_troops[1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    chex.assert_equal(state.board.bases[1, 2], False)

    # check that board is updated correctly
    # with bonus troops
    assert state.board.player_1_troops[0, 0] == 5
    assert state.board.player_1_troops[0, 2] == 4
    assert state.board.player_2_troops[1, 1] == 3
    assert state.board.player_2_troops[2, 2] == 9
    assert state.board.neutral_troops[2, 3] == 3
    assert state.board.neutral_troops[3, 2] == 6


def create_test_state(p1, p2, neutral, bases, time=5):
    """
    Helper function to create an EnvState from lists.
    p1, p2, neutral: 2D lists of ints (4x4) for player1, player2, and neutral troops.
    bases: 2D list of booleans (4x4) indicating base positions.
    """
    board = Board(
        player_1_troops=jnp.array(p1, dtype=jnp.int32),
        player_2_troops=jnp.array(p2, dtype=jnp.int32),
        neutral_troops=jnp.array(neutral, dtype=jnp.int32),
        bases=jnp.array(bases, dtype=bool),
    )
    return EnvState(board=board, time=time)


def test_move_p1_vs_empty():
    """
    p1 moves into an empty tile.
    Setup: p1 has 5 troops at (1,1). The target at (2,1) is empty.
    Expected: p1's source becomes 1 and the target cell gets (5-1)=4 troops.
    """
    # Create a 4x4 board: all zeros except source.
    p1 = [
        [0, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    # Move right from (1,1) to (2,1); action==1 means x+1.
    new_state = move(state, player=0, x=1, y=1, action=1)
    new_p1 = new_state.board.player_1_troops

    # According to the code, if target is empty the source loses troops and becomes 1.
    # The target cell gets the moved troops (5-1).
    print(new_p1)
    assert new_p1[1, 1] == 1, "Source cell should be reduced to 1."
    assert new_p1[1, 2] == 4, "Target cell should have 4 troops (5-1)."


def test_move_p1_vs_p2():
    """
    p1 attacks a tile occupied by p2.
    Setup: p1 has 5 troops at (1,1); p2 has 3 troops at (2,1).
    Expected (per the move logic): since source troops (5-1=4) exceed the target (3),
    the opponent's troops become 0 and the source is reduced.
    """
    p1 = [
        [0, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    # p1 moves right from (1,1) to (2,1).
    new_state = move(state, player=0, x=1, y=1, action=1)
    new_p1 = new_state.board.player_1_troops
    new_p2 = new_state.board.player_2_troops

    # In the "else" branch of move():
    # source troops = 5-1 = 4 and target p2 troops = 3,
    # so p2's target cell becomes 0 and p1's source becomes (4-3)=1.
    assert new_p1[1, 1] == 1, "Source should be reduced to 1 after combat."
    assert new_p1[1, 2] == 1, "Target cell should have 1 troop after combat."
    assert new_p2[1, 2] == 0, "Opponent's troops should be reduced to 0."


def test_move_p1_vs_neutral():
    """
    p1 attacks a tile with neutral troops.
    Setup: p1 has 5 troops at (1,1); a neutral force of 3 is at (1,2).
    Expected: similar to p1 vs p2, the neutral troops are removed.
    """
    p1 = [
        [0, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
    ]
    bases = [[False] * 4 for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    # p1 moves down from (1,1) to (1,2); action==2 means y+1.
    new_state = move(state, player=0, x=1, y=1, action=2)
    new_p1 = new_state.board.player_1_troops
    new_neutral = new_state.board.neutral_troops

    # Outcome: neutral at target becomes 0, and source is reduced.
    assert new_p1[1, 1] == 1, "Source should be reduced to 1 after attacking neutral."
    assert new_neutral[2, 1] == 0, "Neutral troops at target should be eliminated."


def test_move_out_of_bounds():
    """
    Test that if the move would go out-of-bounds, the state remains unchanged.
    Setup: p1 with troops at (0,0); attempt to move left (action 3) which is invalid.
    """
    p1 = [[0] * 4 for _ in range(4)]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    p1[0][0] = 5  # place 5 troops at (0,0)
    state = create_test_state(p1, p2, neutral, bases)
    new_state = move(state, player=0, x=0, y=0, action=3)
    # Expect state to remain unchanged.
    assert (
        new_state.board.player_1_troops == state.board.player_1_troops
    ).all(), "Board should be unchanged when move is out-of-bounds."


def test_move_insufficient_troops():
    """
    Test that if the source tile has fewer than 2 troops, no move is made.
    Setup: p1 with only 1 troop at (1,1).
    """
    p1 = [[0] * 4 for _ in range(4)]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    p1[1][1] = 1  # only 1 troop available
    state = create_test_state(p1, p2, neutral, bases)
    new_state = move(state, player=0, x=1, y=1, action=1)
    # Since there aren’t enough troops, state remains unchanged.
    assert (
        new_state.board.player_1_troops == state.board.player_1_troops
    ).all(), "No move should occur with insufficient troops."


def test_failed_neutral_attack_no_player_troop_transfer():
    """
    Test that a failed attack on a neutral tile does not result in any player's troop
    being placed in the target and that the neutral troop count is reduced.

    Setup:
      - p1 has 5 troops at (1,1) so available attacking troops are 5-1=4.
      - The neutral tile at (1,2) has 7 troops (7 > 4), so the attack fails.

    Expected:
      - p1's source tile (1,1) becomes 1.
      - p1's troops at the target (1,2) remain 0.
      - The neutral troop count at (1,2) reduced to 3.
    """
    # Create a 4x4 board.
    p1 = [
        [0, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 7, 0, 0],  # Neutral troops at (1,2)
        [0, 0, 0, 0],
    ]
    bases = [[False] * 4 for _ in range(4)]

    # Use your helper function to create an EnvState.
    state = create_test_state(p1, p2, neutral, bases)

    # Player 1 moves down (action 2) from (1,1) to (1,2)
    new_state = move(state, player=0, x=1, y=1, action=2)

    new_p1 = new_state.board.player_1_troops
    new_neutral = new_state.board.neutral_troops

    # The source tile should now have 1 troop.
    assert (
        new_p1[1, 1] == 1
    ), "After a failed attack, the source cell should be reduced to 1."

    # The target cell should not contain any player 1 troops.
    assert (
        new_p1[2, 1] == 0
    ), "No player's troops should be present in the target tile after a failed attack on a neutral target."

    # The neutral troop count reduced.
    assert new_neutral[2, 1] == 3, "Neutral troops reduced to 3."
