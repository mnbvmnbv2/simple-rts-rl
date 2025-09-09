import jax.numpy as jnp
import pytest

from src.rts.config import RewardConfig
from src.rts.env import Board, EnvState, reward_function


@pytest.fixture
def reward_config():
    return RewardConfig()


def create_test_state(p1, p2, neutral, bases, time=5):
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
    return EnvState(board=board, time=jnp.array(time, dtype=jnp.int32))


def test_reward_capture_tile(reward_config: RewardConfig):
    """
    Test +1 reward for each captured tile.
    Player 1 captures one new tile.
    """
    # Initial state: player 1 (index 0) has one tile at (0,0)
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 now has two tiles at (0,0) and (0,1)
    p1_final = [
        [1, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]

    state = create_test_state(p1_initial, p2, neutral, bases)
    next_state = create_test_state(p1_final, p2, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    assert reward == 1, "Expected +1 reward for capturing one new tile"


def test_reward_lose_tile(reward_config: RewardConfig):
    """
    Test -1 penalty for each lost tile.
    Player 1 loses one tile.
    """
    # Initial state: player 1 has two tiles
    p1_initial = [
        [3, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 now has only one tile
    p1_final = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]

    state = create_test_state(p1_initial, p2, neutral, bases)
    next_state = create_test_state(p1_final, p2, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    assert reward == -1, "Expected -1 penalty for losing one tile"


def test_reward_capture_base(reward_config: RewardConfig):
    """
    Test +10 reward for capturing a base.
    """
    # Initial state: player 1 has a tile but none is a base
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 gains a new tile at (0,1) which is a base
    p1_final = [
        [1, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

    state = create_test_state(p1_initial, p2, neutral, bases)
    next_state = create_test_state(p1_final, p2, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    # Expect +1 for the new tile and +10 for capturing a base
    assert reward == 11, "Expected +11 reward for capturing a tile with a base"


def test_reward_lose_base(reward_config: RewardConfig):
    """
    Test -10 penalty for losing a base.
    """
    # Initial state: player 1 has two tiles, one is a base
    p1_initial = [
        [3, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 loses the base (only one tile remains)
    p1_final = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2 = [[0] * 4 for _ in range(4)]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

    state = create_test_state(p1_initial, p2, neutral, bases)
    next_state = create_test_state(p1_final, p2, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    # Expected: -1 for the lost tile and -10 for losing the base
    assert reward == -11, "Expected -11 penalty for losing a tile with a base"


def test_reward_defeat_opponent(reward_config: RewardConfig):
    """
    Test +100 reward for defeating the opponent.
    """
    # Initial state: both players have some tiles
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_initial = [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: opponent (player 2) has no tiles left
    p1_final = [
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_final = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]

    state = create_test_state(p1_initial, p2_initial, neutral, bases)
    next_state = create_test_state(p1_final, p2_final, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    # +1 for the new tile and +100 for defeating the opponent
    assert reward == 101, (
        "Expected +101 reward for defeating opponent and capturing new tile"
    )


def test_reward_player_defeated(reward_config: RewardConfig):
    """
    Test -100 penalty for being defeated.
    """
    # Initial state: both players have tiles
    p1_initial = [
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_initial = [
        [0, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 loses all tiles
    p1_final = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_final = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [[False] * 4 for _ in range(4)]

    state = create_test_state(p1_initial, p2_initial, neutral, bases)
    next_state = create_test_state(p1_final, p2_final, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    # -1 for lost tile and -100 penalty for being defeated
    assert reward == -101, "Expected -101 penalty for being defeated"


def test_reward_complex_scenario(reward_config: RewardConfig):
    """
    Test a complex scenario with multiple rewards and penalties.
    Player 1 gains two tiles (one is a base) but loses one tile.
    """
    # Initial state
    p1_initial = [
        [3, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_initial = [
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 1 gains (0,1) with base and (1,1) but loses (0,2)
    p1_final = [
        [2, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_final = [
        [0, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [
        [False, True, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

    state = create_test_state(p1_initial, p2_initial, neutral, bases)
    next_state = create_test_state(p1_final, p2_final, neutral, bases)
    reward = reward_function(state, next_state, player=0, config=reward_config)
    # Expected: +2 for two new tiles, -1 for one lost tile, and +10 for the captured base
    assert reward == 11, "Expected +11 reward for complex scenario"


def test_reward_no_change(reward_config: RewardConfig):
    """
    Test no reward when there's no change in state.
    """
    p1 = [
        [3, 0, 0, 0],
        [0, 2, 0, 0],
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
    reward = reward_function(state, state, player=0, config=reward_config)
    assert reward == 0, "Expected 0 reward when state doesn't change"


def test_reward_for_player_two(reward_config: RewardConfig):
    """
    Test rewards are calculated correctly for player 2.
    """
    # Initial state
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_initial = [
        [0, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: player 2 captures a base from player 1
    p1_final = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    p2_final = [
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    neutral = [[0] * 4 for _ in range(4)]
    bases = [
        [True, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

    state = create_test_state(p1_initial, p2_initial, neutral, bases)
    next_state = create_test_state(p1_final, p2_final, neutral, bases)
    reward = reward_function(state, next_state, player=1, config=reward_config)
    # Expect: +1 for the new tile, +10 for the captured base, and +100 for victory
    assert reward == 111, "Expected +111 reward for player 2"
