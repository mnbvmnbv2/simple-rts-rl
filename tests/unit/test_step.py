import chex
import jax.numpy as jnp
import pytest

from src.rts.config import EnvConfig
from src.rts.env import Board, EnvState, is_done, move, reinforce_troops
from tests.helpers import assert_valid_state


@pytest.fixture
def config():
    return EnvConfig(
        num_players=2,
        board_width=4,
        board_height=4,
        num_neutral_bases=0,
        num_neutral_troops_start=0,
        neutral_troops_min=0,
        neutral_troops_max=0,
        player_start_troops=5,
        bonus_time=10,
    )


@pytest.fixture
def board():
    # Create separate 4x4 arrays for player 1 and player 2
    p1 = jnp.zeros((4, 4), dtype=jnp.int32)
    p2 = jnp.zeros((4, 4), dtype=jnp.int32)
    neutral_troops = jnp.zeros((4, 4), dtype=jnp.int32)
    bases = jnp.zeros((4, 4), dtype=jnp.bool_)

    # p1 troops at (0,0) and (0,2)
    p1 = p1.at[0, 0].set(4)
    p1 = p1.at[0, 2].set(2)
    # p2 troops at (1,1) and (2,2)
    p2 = p2.at[1, 1].set(1)
    p2 = p2.at[2, 2].set(8)
    # neutral troops at (2,3) and (3,2)
    neutral_troops = neutral_troops.at[2, 3].set(3)
    neutral_troops = neutral_troops.at[3, 2].set(6)
    # bases at (0,2), (1,1), and (3,2)
    bases = bases.at[0, 2].set(True)
    bases = bases.at[1, 1].set(True)
    bases = bases.at[3, 2].set(True)

    board = Board(
        player_troops=jnp.stack([p1, p2], axis=0),
        neutral_troops=neutral_troops,
        bases=bases,
    )
    return board


def create_test_state(p1, p2, neutral, bases, time=5):
    """
    Helper function to create an EnvState given 2D lists for p1, p2, neutral troops and bases.
    Combines p1 and p2 into a single `player_troops` array.
    """
    board = Board(
        player_troops=jnp.stack(
            [jnp.array(p1, dtype=jnp.int32), jnp.array(p2, dtype=jnp.int32)], axis=0
        ),
        neutral_troops=jnp.array(neutral, dtype=jnp.int32),
        bases=jnp.array(bases, dtype=bool),
    )
    return EnvState(board=board, time=jnp.array(time, dtype=jnp.int32))


def test_increase_troops(board: jnp.array, config: EnvConfig):
    state = EnvState(board=board, time=jnp.array(4, dtype=jnp.int32))
    state = reinforce_troops(state, config)
    assert_valid_state(state)

    # Check a couple of blank tiles remain unchanged.
    # Access player 1 via state.board.player_troops[0] and player 2 via [1]
    assert state.board.player_troops[0, 0, 1] == 0
    assert state.board.player_troops[1, 0, 1] == 0
    assert state.board.neutral_troops[0, 1] == 0
    chex.assert_equal(state.board.bases[0, 1], False)

    assert state.board.player_troops[0, 1, 2] == 0
    assert state.board.player_troops[1, 1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    chex.assert_equal(state.board.bases[1, 2], False)

    # Verify that the board is updated correctly (no bonus troops added, since time != 0).
    assert state.board.player_troops[0, 0, 0] == 4
    # For player 1 at (0,2), reinforcement should update troop count; expected value is 3.
    assert state.board.player_troops[0, 0, 2] == 3
    # For player 2: (1,1) should have increased from 1 to 2 and (2,2) remains 8.
    assert state.board.player_troops[1, 1, 1] == 2
    assert state.board.player_troops[1, 2, 2] == 8
    assert state.board.neutral_troops[2, 3] == 3
    assert state.board.neutral_troops[3, 2] == 6


def test_increase_troops_bonus(board: jnp.array, config: EnvConfig):
    state = EnvState(board=board, time=jnp.array(0, dtype=jnp.int32))
    state = reinforce_troops(state, config)
    assert_valid_state(state)

    # Check that blank tiles remain blank.
    assert state.board.player_troops[0, 0, 1] == 0
    assert state.board.player_troops[1, 0, 1] == 0
    assert state.board.neutral_troops[0, 1] == 0
    chex.assert_equal(state.board.bases[0, 1], False)

    assert state.board.player_troops[0, 1, 2] == 0
    assert state.board.player_troops[1, 1, 2] == 0
    assert state.board.neutral_troops[1, 2] == 0
    chex.assert_equal(state.board.bases[1, 2], False)

    # Verify that bonus troops are added.
    # With bonus (time==0), each active cell gets one extra troop.
    assert state.board.player_troops[0, 0, 0] == 5
    assert state.board.player_troops[0, 0, 2] == 4
    assert state.board.player_troops[1, 1, 1] == 3
    assert state.board.player_troops[1, 2, 2] == 9
    assert state.board.neutral_troops[2, 3] == 3
    assert state.board.neutral_troops[3, 2] == 6


def test_move_p1_vs_empty():
    """
    p1 moves into an empty tile.
    Setup: player 1 has 5 troops at (1,1). The target at (1,2) is empty.
    Expected: The source becomes 1 and the target gets (5-1)=4 troops.
    """
    # Create a 4x4 board: all cells empty except for the source cell.
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
    # Move right (action==1 means x+1) from (1,1) to (1,2) for player 1.
    new_state = move(state, player=0, x=1, y=1, action=1)
    new_p1 = new_state.board.player_troops[0]
    # Verify outcome.
    assert new_p1[1, 1] == 1, "Source cell should be reduced to 1."
    assert new_p1[1, 2] == 4, "Target cell should have 4 troops (5-1)."


def test_move_p1_vs_p2():
    """
    p1 attacks a tile occupied by player 2.
    Setup: p1 has 5 troops at (1,1); p2 has 3 troops at (1,2).
    Expected: Since attacking troops (5-1=4) exceed p2's 3, player 2 loses troops at (1,2)
              and player 1 ends up with remaining troops.
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
    # p1 moves right (from (1,1) to (1,2)).
    new_state = move(state, player=0, x=1, y=1, action=1)
    new_p1 = new_state.board.player_troops[0]
    new_p2 = new_state.board.player_troops[1]
    # In this combat, player 1's source becomes 1; player 2 should have 0 at (1,2)
    # and player 1 gets remaining survivors.
    assert new_p1[1, 1] == 1, "Source should be reduced to 1 after combat."
    assert new_p1[1, 2] == 1, (
        "Player 1 should place surviving troops (if any) at target."
    )
    assert new_p2[1, 2] == 0, "Player 2's troops should be reduced to 0."


def test_move_p1_vs_neutral():
    """
    p1 attacks a tile occupied by neutral troops.
    Setup: p1 has 5 troops at (1,1); neutral cell at (2,1) has 3 troops.
    Expected: Neutral troops are removed and p1's source cell is reduced.
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
    # p1 moves down (action==2 means y+1) from (1,1) to (2,1)
    new_state = move(state, player=0, x=1, y=1, action=2)
    new_p1 = new_state.board.player_troops[0]
    new_neutral = new_state.board.neutral_troops
    assert new_p1[1, 1] == 1, "Source should be reduced to 1 after attacking neutral."
    assert new_neutral[2, 1] == 0, "Neutral troops at target should be eliminated."


def test_move_out_of_bounds():
    """
    Test that if a move goes out-of-bounds, the state remains unchanged.
    Setup: p1 has troops at (0,0); attempt to move left (action==3) which is invalid.
    """
    p1 = [[5, 0, 0, 0] for _ in range(4)]
    p2 = [[0, 0, 0, 0] for _ in range(4)]
    neutral = [[0, 0, 0, 0] for _ in range(4)]
    bases = [[False, False, False, False] for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    new_state = move(state, player=0, x=0, y=0, action=3)
    # Verify that the state remains unchanged.
    assert jnp.all(new_state.board.player_troops[0] == state.board.player_troops[0])
    assert jnp.all(new_state.board.player_troops[1] == state.board.player_troops[1])
    assert jnp.all(new_state.board.neutral_troops == state.board.neutral_troops)


def test_move_insufficient_troops():
    """
    Test that no move occurs if the source tile has fewer than 2 troops.
    Setup: p1 has only 1 troop at (1,1).
    """
    p1 = [[0, 0, 0, 0] for _ in range(4)]
    p2 = [[0, 0, 0, 0] for _ in range(4)]
    neutral = [[0, 0, 0, 0] for _ in range(4)]
    bases = [[False, False, False, False] for _ in range(4)]
    p1[1][1] = 1  # Only 1 troop available.
    state = create_test_state(p1, p2, neutral, bases)
    new_state = move(state, player=0, x=1, y=1, action=1)
    # Since there aren’t enough troops, the state should remain unchanged.
    assert jnp.all(new_state.board.player_troops[0] == state.board.player_troops[0])
    assert jnp.all(new_state.board.player_troops[1] == state.board.player_troops[1])
    assert jnp.all(new_state.board.neutral_troops == state.board.neutral_troops)


def test_failed_neutral_attack_no_player_troop_transfer():
    """
    Test that a failed attack on a neutral tile (when attacking troops are less than neutral troops)
    does not transfer any player troops to the target.
    Setup:
      - p1 has 5 troops at (1,1) so available attacking troops = 4.
      - The neutral cell at (2,1) has 7 troops (7 > 4), so the attack fails.
    Expected:
      - p1's source becomes 1.
      - The target cell remains without player troops.
      - The neutral troop count at (2,1) is reduced appropriately.
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
        [0, 7, 0, 0],  # Neutral troops at (2,1)
        [0, 0, 0, 0],
    ]
    bases = [[False, False, False, False] for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    new_state = move(state, player=0, x=1, y=1, action=2)
    new_p1 = new_state.board.player_troops[0]
    new_neutral = new_state.board.neutral_troops
    assert new_p1[1, 1] == 1, "After a failed attack, source should reduce to 1."
    assert new_p1[2, 1] == 0, "Target cell should not receive any player troops."
    # Expect the neutral troops to be reduced (by the damage amount).
    # In this scenario, damage = min(4, 7) = 4 so neutral troops become 7-4 = 3.
    assert new_neutral[2, 1] == 3, "Neutral troops should be reduced to 3."


def test_move_equal_combat():
    """
    Test a move where attacking troops exactly equal total defenders.
    In this case, the attack reduces the source to 1 and leaves target empty.
    """
    # p1 at (1,1) has 5 troops, so attacking troops = 4.
    # p2 at (1,2) has exactly 4 troops.
    p1 = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    p2 = [
        [0, 0, 0],
        [0, 0, 4],
        [0, 0, 0],
    ]
    neutral = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    bases = [
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ]
    state = create_test_state(p1, p2, neutral, bases)
    # Action 1 moves from (1,1) to (1,2)
    new_state = move(state, player=0, x=1, y=1, action=1)
    new_p1 = new_state.board.player_troops[0]
    new_p2 = new_state.board.player_troops[1]
    assert new_p1[1, 1] == 1, "Source should be reduced to 1."
    assert new_p1[1, 2] == 0, "Target should remain 0 after equal combat."
    assert new_p2[1, 2] == 0, "Opponent's troops should be reduced to 0."


def test_move_invalid_action_no_change():
    """
    Test that an invalid move action (e.g. moving up from the top edge) leaves the state unchanged.
    """
    p1 = [[5, 0], [0, 0]]
    p2 = [[0, 0], [0, 0]]
    neutral = [[0, 0], [0, 0]]
    bases = [[True, False], [False, False]]
    state = create_test_state(p1, p2, neutral, bases)
    # For cell (0,0), only right and down moves are legal.
    # Provide invalid action 0 (up).
    new_state = move(state, player=0, x=0, y=0, action=0)
    # Check that player 1's grid remains unchanged.
    assert jnp.all(new_state.board.player_troops[0] == state.board.player_troops[0])
    assert jnp.all(new_state.board.player_troops[1] == state.board.player_troops[1])
    assert jnp.all(new_state.board.neutral_troops == state.board.neutral_troops)


def test_move_own_troops():
    """
    Test a move onto a cell where the same player already has troops.
    """
    # p1 at (1,1) has 5 troops, attacking troops = 4.
    # p1 at (0,1) has 7 troops.
    p1 = [
        [0, 7, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    p2 = [
        [0, 0, 0],
        [0, 0, 4],
        [0, 0, 0],
    ]
    neutral = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    bases = [
        [False, False, False],
        [False, True, False],
        [False, False, False],
    ]
    state = create_test_state(p1, p2, neutral, bases)
    # Move up (action==0) from (1,1) to (0,1)
    new_state = move(state, player=0, x=1, y=1, action=0)
    new_p1 = new_state.board.player_troops[0]
    assert new_p1[1, 1] == 1, "Source should be reduced to 1."
    # The target now contains the sum of existing and surviving attacking troops.
    assert new_p1[0, 1] == 11, "Target cell should have combined troops."


def test_is_done_game_not_over():
    """
    Verify that is_done returns False when both players have troops.
    """
    p1 = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    assert not is_done(state), "Game should not be done when both players have troops."


def test_is_done_game_over():
    """
    Verify that is_done returns True when one player has no troops.
    """
    p1 = [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]
    state = create_test_state(p1, p2, neutral, bases)
    assert is_done(state), "Game should be done when one player has no troops."
