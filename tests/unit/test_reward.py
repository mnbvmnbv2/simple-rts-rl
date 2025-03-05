import jax.numpy as jnp
from src.rts.env import Board, EnvState, reward_function


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


def test_reward_capture_tile():
    """
    Test +1 reward for each captured tile.
    Player 1 captures one new tile.
    """
    # Initial state: p1 has one tile at (0,0)
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: p1 has two tiles at (0,0) and (0,1)
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

    reward = reward_function(state, next_state, player=0)
    assert reward == 1, "Expected +1 reward for capturing one new tile"


def test_reward_lose_tile():
    """
    Test -1 penalty for each lost tile.
    Player 1 loses one tile.
    """
    # Initial state: p1 has two tiles
    p1_initial = [
        [3, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: p1 has only one tile
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

    reward = reward_function(state, next_state, player=0)
    assert reward == -1, "Expected -1 penalty for losing one tile"


def test_reward_capture_base():
    """
    Test +10 reward for capturing a base.
    """
    # Initial state: p1 has one tile, no base
    p1_initial = [
        [3, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: p1 has two tiles, one is a base
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

    reward = reward_function(state, next_state, player=0)
    # +1 for new tile, +10 for capturing base
    assert reward == 11, "Expected +11 reward for capturing a tile with a base"


def test_reward_lose_base():
    """
    Test -10 penalty for losing a base.
    """
    # Initial state: p1 has two tiles, one is a base
    p1_initial = [
        [3, 2, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    # Final state: p1 has one tile, no base
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

    reward = reward_function(state, next_state, player=0)
    # -1 for lost tile, -10 for lost base
    assert reward == -11, "Expected -11 penalty for losing a tile with a base"


def test_reward_defeat_opponent():
    """
    Test +100 reward for defeating the opponent.
    """
    # Initial state: p1 and p2 each have tiles
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

    # Final state: p2 has no tiles left
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

    reward = reward_function(state, next_state, player=0)
    # +1 for new tile, +100 for victory
    assert (
        reward == 101
    ), "Expected +101 reward for defeating opponent and capturing new tile"


def test_reward_player_defeated():
    """
    Test -100 penalty for being defeated.
    """
    # Initial state: p1 and p2 each have tiles
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

    # Final state: p1 has no tiles left
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

    reward = reward_function(state, next_state, player=0)
    # -1 for lost tile, -100 for defeat
    assert reward == -101, "Expected -101 penalty for being defeated"


def test_reward_complex_scenario():
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

    # Final state: p1 gained (0,1) with base, gained (1,1) but lost (0,2)
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

    reward = reward_function(state, next_state, player=0)
    # +2 for two new tiles, -1 for one lost tile, +10 for captured base
    assert reward == 11, "Expected +11 reward for complex scenario"


def test_reward_no_change():
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

    reward = reward_function(state, state, player=0)
    assert reward == 0, "Expected 0 reward when state doesn't change"


def test_reward_for_player_two():
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

    # Final state: p2 captured a base from p1
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

    reward = reward_function(state, next_state, player=1)
    # +1 for new tile, +10 for captured base, +100 for victory
    assert reward == 111, "Expected +111 reward for player 2"
