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


def init_state(rng_key: jnp.ndarray, params: EnvConfig) -> EnvState:
    """Each tile has 4 channels:
    1. Player 1 troops
    2. Player 2 troops
    3. Neutral troops
    4. Base"""
    # create a board
    width = params.board_width
    height = params.board_height

    player_1_troops = jnp.zeros((width, height), dtype=jnp.int32)
    player_2_troops = jnp.zeros((width, height), dtype=jnp.int32)
    neutral_troops = jnp.zeros((width, height), dtype=jnp.int32)
    bases = jnp.zeros((width, height), dtype=jnp.bool_)
    # randomly select 2 start positions that should be unique
    pos1 = jax.random.randint(rng_key, (2,), 0, width)
    rng_key, _ = jax.random.split(rng_key)
    pos2 = jax.random.randint(rng_key, (2,), 0, width)
    while jnp.array_equal(pos1, pos2):
        rng_key, _ = jax.random.split(rng_key)
        pos2 = jax.random.randint(rng_key, (2,), 0, width)

    # set p1 troop and base
    player_1_troops = player_1_troops.at[pos1[0], pos1[1]].set(5)
    bases = bases.at[pos1[0], pos1[1]].set(True)
    # set p2 troop and base
    player_2_troops = player_2_troops.at[pos2[0], pos2[1]].set(5)
    bases = bases.at[pos2[0], pos2[1]].set(True)

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
        neutral_troops = neutral_troops.at[pos[0], pos[1]].set(num_troops)
        bases = bases.at[pos[0], pos[1]].set(True)

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
        neutral_troops = neutral_troops.at[pos[0], pos[1]].set(num_troops)

    board = Board(
        player_1_troops=player_1_troops,
        player_2_troops=player_2_troops,
        neutral_troops=neutral_troops,
        bases=bases,
    )

    return EnvState(board=board)


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
        troops = troops.at[target_y, target_x].set(
            troops[y, x] - 1 + troops[target_y, target_x]
        )
        troops = troops.at[y, x].set(1)
    elif target_troops > sorce_troops:
        opponent_troops = opponent_troops.at[target_y, target_x].set(
            target_troops - sorce_troops + 1
        )
        troops = troops.at[y, x].set(1)
    else:
        opponent_troops = opponent_troops.at[target_y, target_x].set(0)
        troops = troops.at[y, x].set(sorce_troops - target_troops)
        if troops[y, x] > 1:
            troops = troops.at[target_y, target_x].set(troops[y, x] - 1)
            troops = troops.at[y, x].set(1)

    return EnvState(board=board, time=state.time)


@jax.jit
def increase_troops(state: EnvState) -> EnvState:
    # We only increase troops for player 1 and player 2
    board = state.board
    bonus_troops = (state.time == 0).astype(int)

    p1_troop_locations = board.player_1_troops > 0
    p2_troop_locations = board.player_2_troops > 0
    base_bonus = board.bases.astype(jnp.int32)

    new_player_1 = (
        board.player_1_troops
        + bonus_troops * p1_troop_locations.astype(jnp.int32)
        + base_bonus * p1_troop_locations.astype(jnp.int32)
    )
    new_player_2 = (
        board.player_2_troops
        + bonus_troops * p2_troop_locations.astype(jnp.int32)
        + base_bonus * p2_troop_locations.astype(jnp.int32)
    )

    new_board = board.replace(
        player_1_troops=new_player_1, player_2_troops=new_player_2
    )

    # Decrese time and increase to 10 if bonus troops
    time = state.time - 1 + bonus_troops * 10
    return EnvState(board=new_board, time=time)
