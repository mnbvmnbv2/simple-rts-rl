import jax
import jax.numpy as jnp
from flax import struct

from src.rts.config import EnvConfig


@struct.dataclass
class EnvState:
    board: jnp.ndarray
    time: int = 5


def move(state: EnvState, player: int, x: int, y: int, action: int) -> EnvState:
    board = state.board
    if board.shape[0] <= x or board.shape[1] <= y:
        return state
    if board[y, x, player] < 2:
        return state
    target_x, target_y = x, y
    if action == 0:
        target_y = y - 1
    elif action == 1:
        target_x = x + 1
    elif action == 2:
        target_y = y + 1
    elif action == 3:
        target_x = x - 1

    # Check if the target is within bounds
    within_x = target_x >= 0 and target_x < board.shape[1]
    within_y = target_y >= 0 and target_y < board.shape[0]
    if not within_x or not within_y:
        return state

    # Check if the target has opponent troops
    if board[target_y, target_x, (player + 1) % 2] > 0:
        target_troops = board[target_y, target_x, (player + 1) % 2]
        opponent = (player + 1) % 2
    # Check if the target has neutral troops
    elif board[target_y, target_x, 2] > 0:
        target_troops = board[target_y, target_x, 2 % 2]
        opponent = 2
    else:
        target_troops = 0
        opponent = None

    sorce_troops = board[y, x, player]
    if opponent is None:
        board = board.at[target_y, target_x, player].set(
            board[y, x, player] - 1 + board[target_y, target_x, player]
        )
        board = board.at[y, x, player].set(1)
    elif target_troops > sorce_troops:
        board = board.at[target_y, target_x, opponent].set(
            target_troops - sorce_troops + 1
        )
        board = board.at[y, x, player].set(1)
    else:
        board = board.at[target_y, target_x, opponent].set(0)
        board = board.at[y, x, player].set(sorce_troops - target_troops)
        if board[y, x, player] > 1:
            board = board.at[target_y, target_x, player].set(board[y, x, player] - 1)
            board = board.at[y, x, player].set(1)

    return EnvState(board=board, time=state.time)


def init_state(rng_key: jnp.ndarray, params: EnvConfig) -> EnvState:
    """Each tile has 4 channels:
    1. Player 1 troops
    2. Player 2 troops
    3. Neutral troops
    4. Base"""
    # create a board
    width = params.board_width
    height = params.board_height

    board = jnp.zeros((width, height, 4), dtype=jnp.int32)
    # randomly select 2 start positions that should be unique
    pos1 = jax.random.randint(rng_key, (2,), 0, width)
    rng_key, _ = jax.random.split(rng_key)
    pos2 = jax.random.randint(rng_key, (2,), 0, width)
    while jnp.array_equal(pos1, pos2):
        rng_key, _ = jax.random.split(rng_key)
        pos2 = jax.random.randint(rng_key, (2,), 0, width)

    # set p1 troop and base
    board = board.at[pos1[0], pos1[1], 0].set(5)
    board = board.at[pos1[0], pos1[1], 3].set(1)
    # set p2 troop and base
    board = board.at[pos2[0], pos2[1], 1].set(5)
    board = board.at[pos2[0], pos2[1], 3].set(1)

    # set random neutral bases
    for i in range(params.num_neutral_bases):
        rng_key, _ = jax.random.split(rng_key)
        pos = jax.random.randint(rng_key, (2,), 0, width)
        while jnp.array_equal(pos, pos1) or jnp.array_equal(pos, pos2):
            rng_key, _ = jax.random.split(rng_key)
            pos = jax.random.randint(rng_key, (2,), 0, width)
        # set random number of neutral troops
        rng_key, _ = jax.random.split(rng_key)
        num_troops = jax.random.randint(
            rng_key,
            (),
            params.neutral_bases_min_troops,
            params.neutral_bases_max_troops,
        )
        board = board.at[pos[0], pos[1], 2].set(num_troops)
        board = board.at[pos[0], pos[1], 3].set(1)

    # set random neutral troops
    for i in range(params.num_neutral_troops_start):
        rng_key, _ = jax.random.split(rng_key)
        pos = jax.random.randint(rng_key, (2,), 0, width)
        while jnp.array_equal(pos, pos1) or jnp.array_equal(pos, pos2):
            rng_key, _ = jax.random.split(rng_key)
            pos = jax.random.randint(rng_key, (2,), 0, width)
        # set random number of neutral troops
        rng_key, _ = jax.random.split(rng_key)
        num_troops = jax.random.randint(
            rng_key,
            shape=(),
            minval=params.neutral_bases_min_troops,
            maxval=params.neutral_bases_max_troops,
        )
        board = board.at[pos[0], pos[1], 2].set(num_troops)

    return EnvState(board=board)


def increase_troops(state: EnvState) -> EnvState:
    # We only increase troops for player 1 and player 2
    board = state.board
    bonus_troops = state.time == 0
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            for k in range(2):
                # Increase troops for all places with troops if bonus troops
                if board[i, j, k] > 0:
                    board = board.at[i, j, k].set(board[i, j, k] + bonus_troops)
                    # Increse troops for all bases
                    if board[i, j, 3] > 0:
                        board = board.at[i, j, k].set(board[i, j, k] + 1)
    # Decrese time and increase to 10 if bonus troops
    time = state.time - 1 + bonus_troops * 10
    return EnvState(board=board, time=time)
