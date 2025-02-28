import jax
import jax.numpy as jnp
from flax import struct

from src.rts.config import EnvConfig


@struct.dataclass
class Board:
    player_1_troops: jnp.ndarray  # shape: (width, height) of int
    player_2_troops: jnp.ndarray  # shape: (width, height) of int
    neutral_troops: jnp.ndarray  # shape: (width, height) of int
    bases: jnp.ndarray  # shape: (width, height) of bool

    @property
    def width(self) -> int:
        return self.player_1_troops.shape[0]

    @property
    def height(self) -> int:
        return self.player_1_troops.shape[1]


@struct.dataclass
class EnvState:
    board: Board
    time: int = 5


def move(state: EnvState, player: int, x: int, y: int, action: int) -> EnvState:
    board = state.board
    if board.width <= x or board.height <= y:
        return state

    if player == 0:
        troops = board.player_1_troops
    else:
        troops = board.player_2_troops

    if troops[y, x] < 2:
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
    if (
        target_x < 0
        or target_y < 0
        or target_x >= board.width
        or target_y >= board.height
    ):
        return state

    if player == 0:
        opponent_troops = board.player_2_troops
    else:
        opponent_troops = board.player_1_troops

    # Check if the target
    # has other player's troops
    if opponent_troops[target_y, target_x] > 0:
        target_troops = opponent_troops[target_y, target_x]
    # Check if the target has neutral troops
    elif board.neutral_troops[target_y, target_x] > 0:
        target_troops = board.neutral_troops[target_y, target_x]
        opponent_troops = board.neutral_troops
    else:
        target_troops = 0

    sorce_troops = troops[y, x] - 1
    if target_troops == 0:
        board = troops.at[target_y, target_x].set(
            troops[y, x] - 1 + troops[target_y, target_x]
        )
        board = troops.at[y, x].set(1)
    elif target_troops > sorce_troops:
        board = opponent_troops.at[target_y, target_x].set(
            target_troops - sorce_troops + 1
        )
        board = troops.at[y, x].set(1)
    else:
        board = opponent_troops.at[target_y, target_x].set(0)
        board = troops.at[y, x].set(sorce_troops - target_troops)
        if troops[y, x] > 1:
            board = troops.at[target_y, target_x].set(troops[y, x] - 1)
            board = troops.at[y, x].set(1)

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
    for i in range(board.width):
        for j in range(board.height):
            for troops_board in [board.player_1_troops, board.player_2_troops]:
                # Increase troops for all places with troops if bonus troops
                if troops_board[i, j] > 0:
                    troops_board = troops_board.at[i, j].set(
                        troops_board[i, j] + bonus_troops
                    )
                    # Increse troops for all bases
                    if troops_board[i, j, 3] > 0:
                        troops_board = troops_board.at[i, j].set(troops_board[i, j] + 1)
    # Decrese time and increase to 10 if bonus troops
    time = state.time - 1 + bonus_troops * 10
    return EnvState(board=board, time=time)
