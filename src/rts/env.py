from functools import partial

import jax
import jax.numpy as jnp

from src.rts.config import EnvConfig
from src.rts.state import Board, EnvState
from src.rts.utils import get_legal_moves, fixed_argwhere


@partial(jax.jit, static_argnames=("config",))
def init_state(
    rng_key: jnp.ndarray,
    config: EnvConfig,
) -> EnvState:
    """Generate a new environment state using a flat grid approach."""
    width = config.board_width
    height = config.board_height
    total_cells = width * height

    # Total number of special cells to assign.
    num_special = 2 + config.num_neutral_bases + config.num_neutral_troops_start
    assert total_cells >= num_special, "Board too small for the required placements."

    # Get a random permutation of all cell indices.
    rng_key, subkey = jax.random.split(rng_key)
    all_indices = jax.random.permutation(subkey, total_cells)

    # Assign positions (flattened indices) for the special placements.
    p1_index = all_indices[0]  # Player 1 base
    p2_index = all_indices[1]  # Player 2 base
    neutral_base_indices = all_indices[2 : 2 + config.num_neutral_bases]
    neutral_troop_indices = all_indices[
        2 + config.num_neutral_bases : 2
        + config.num_neutral_bases
        + config.num_neutral_troops_start
    ]

    # Initialize flat arrays for the board.
    player_1_troops_flat = jnp.zeros(total_cells, dtype=jnp.int32)
    player_2_troops_flat = jnp.zeros(total_cells, dtype=jnp.int32)
    neutral_troops_flat = jnp.zeros(total_cells, dtype=jnp.int32)
    bases_flat = jnp.zeros(total_cells, dtype=jnp.bool_)

    # Set player bases with troops.
    player_1_troops_flat = player_1_troops_flat.at[p1_index].set(
        config.player_start_troops
    )
    bases_flat = bases_flat.at[p1_index].set(True)
    player_2_troops_flat = player_2_troops_flat.at[p2_index].set(
        config.player_start_troops
    )
    bases_flat = bases_flat.at[p2_index].set(True)

    # For neutral bases, assign a random troop count and mark the base.
    rng_key, subkey = jax.random.split(rng_key)
    neutral_base_troops = jax.random.randint(
        subkey,
        shape=(config.num_neutral_bases,),
        minval=config.neutral_troops_min,
        maxval=config.neutral_troops_max,
    )
    neutral_troops_flat = neutral_troops_flat.at[neutral_base_indices].set(
        neutral_base_troops
    )
    bases_flat = bases_flat.at[neutral_base_indices].set(True)

    # For neutral troops without a base, assign a random troop count.
    rng_key, subkey = jax.random.split(rng_key)
    neutral_troops_start = jax.random.randint(
        subkey,
        shape=(config.num_neutral_troops_start,),
        minval=config.neutral_troops_min,
        maxval=config.neutral_troops_max,
    )
    neutral_troops_flat = neutral_troops_flat.at[neutral_troop_indices].set(
        neutral_troops_start
    )

    # Reshape flat arrays back into (height, width) grids.
    player_1_troops = player_1_troops_flat.reshape(
        (config.board_height, config.board_width)
    )
    player_2_troops = player_2_troops_flat.reshape(
        (config.board_height, config.board_width)
    )
    neutral_troops = neutral_troops_flat.reshape(
        (config.board_height, config.board_width)
    )
    bases = bases_flat.reshape((config.board_height, config.board_width))

    board = Board(
        player_1_troops=player_1_troops,
        player_2_troops=player_2_troops,
        neutral_troops=neutral_troops,
        bases=bases,
    )

    return EnvState(board=board)


@jax.jit
def move(
    state: EnvState,
    player: int,
    x: int,
    y: int,
    action: int,
) -> EnvState:
    board = state.board

    player_troops = jnp.where(player == 0, board.player_1_troops, board.player_2_troops)
    opponent_player_troops = jnp.where(
        player == 0, board.player_2_troops, board.player_1_troops
    )
    neutral_troops = board.neutral_troops

    target_x = jnp.where(action == 1, x + 1, jnp.where(action == 3, x - 1, x))
    target_y = jnp.where(action == 0, y - 1, jnp.where(action == 2, y + 1, y))

    # Check if the move is valid
    source_in_bounds = jnp.logical_and(x >= 0, x < board.width) & jnp.logical_and(
        y >= 0, y < board.height
    )
    target_in_bounds = jnp.logical_and(
        target_y >= 0, target_y < board.height
    ) & jnp.logical_and(target_x >= 0, target_x < board.width)
    has_enough_troops = player_troops[y, x] > 1
    valid_move = source_in_bounds & target_in_bounds & has_enough_troops

    # Check number of opponent troops in target
    num_opponent_player_troops = opponent_player_troops[target_y, target_x]
    num_opponent_neutral_troops = board.neutral_troops[target_y, target_x]
    total_oppoent_troops = num_opponent_player_troops + num_opponent_neutral_troops

    # Battle logic
    num_attacking_troops = player_troops[y, x] - 1
    remaining_attacking_troops = jnp.maximum(
        0, num_attacking_troops - total_oppoent_troops
    )
    num_opponent_player_troops = jnp.maximum(
        0, num_opponent_player_troops - num_attacking_troops
    )
    num_opponent_neutral_troops = jnp.maximum(
        0, num_opponent_neutral_troops - num_attacking_troops
    )

    # Update board
    player_troops_at_target = (
        remaining_attacking_troops + player_troops[target_y, target_x]
    )
    player_troops = player_troops.at[y, x].set(
        jnp.where(valid_move, 1, player_troops[y, x])
    )
    player_troops = player_troops.at[target_y, target_x].set(
        jnp.where(
            valid_move, player_troops_at_target, player_troops[target_y, target_x]
        )
    )
    opponent_player_troops = opponent_player_troops.at[target_y, target_x].set(
        jnp.where(
            valid_move,
            num_opponent_player_troops,
            opponent_player_troops[target_y, target_x],
        )
    )
    neutral_troops = board.neutral_troops.at[target_y, target_x].set(
        jnp.where(
            valid_move,
            num_opponent_neutral_troops,
            board.neutral_troops[target_y, target_x],
        )
    )

    new_board = board.replace(
        player_1_troops=jnp.where(player == 0, player_troops, opponent_player_troops),
        player_2_troops=jnp.where(player == 1, player_troops, opponent_player_troops),
        neutral_troops=neutral_troops,
    )

    return EnvState(board=new_board, time=state.time)


@partial(jax.jit, static_argnames=("config",))
def reinforce_troops(
    state: EnvState,
    config: EnvConfig,
) -> EnvState:
    """This function increases troops for players and updates the time.

    When time is 0, all tiles with player troops get a bonus troop.
    Regardless of time, all tiles with player troops on a base get a troop.
    """
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
    time = state.time - 1 + bonus_troops * config.bonus_time
    return EnvState(board=new_board, time=time)


@jax.jit
def reward_function(
    state: EnvState,
    next_state: EnvState,
    player: int,
) -> jnp.ndarray:
    """
    Rewards:
        - +1 reward if captured new tile
        - +10 reward if captured base
        - +100 reward if defeatued opponent

    Penalties:
        - -1 if lost tile
        - -10 if lost base
        - -100 if defeated
    """
    # Get player and opponent troops arrays
    player_troops_current = jnp.where(
        player == 0, state.board.player_1_troops, state.board.player_2_troops
    )
    player_troops_next = jnp.where(
        player == 0, next_state.board.player_1_troops, next_state.board.player_2_troops
    )

    opponent_troops_current = jnp.where(
        player == 0, state.board.player_2_troops, state.board.player_1_troops
    )
    opponent_troops_next = jnp.where(
        player == 0, next_state.board.player_2_troops, next_state.board.player_1_troops
    )

    # Calculate tile changes
    player_tiles_current = jnp.sum(player_troops_current > 0)
    player_tiles_next = jnp.sum(player_troops_next > 0)
    tiles_change = player_tiles_next - player_tiles_current

    # Calculate base changes
    player_bases_current = jnp.sum((player_troops_current > 0) & state.board.bases)
    player_bases_next = jnp.sum((player_troops_next > 0) & next_state.board.bases)
    bases_change = player_bases_next - player_bases_current

    # Check for victory/defeat
    opponent_tiles_current = jnp.sum(opponent_troops_current > 0)
    opponent_tiles_next = jnp.sum(opponent_troops_next > 0)

    victory = jnp.logical_and(
        opponent_tiles_current > 0, opponent_tiles_next == 0
    ).astype(jnp.int32)
    defeat = jnp.logical_and(player_tiles_current > 0, player_tiles_next == 0).astype(
        jnp.int32
    )

    # Calculate total reward
    total_reward = tiles_change + 10 * bases_change + 100 * victory - 100 * defeat

    return total_reward


@jax.jit
def is_done(state: EnvState) -> bool:
    """Check if the game is finished, when either player has no troops left."""
    return jnp.logical_or(
        jnp.all(state.board.player_1_troops == 0),
        jnp.all(state.board.player_2_troops == 0),
    )


@partial(jax.jit, static_argnames=("config",))
def random_step(
    state: EnvState,
    rng_key: jnp.ndarray,
    config: EnvConfig,
) -> EnvState:
    # p1 move
    legal_actions_mask = get_legal_moves(state, 0)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask, max_actions=state.board.width * state.board.height * 4
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    next_state: EnvState = move(state, 0, action[1], action[0], action[2])
    # p2 move
    legal_actions_mask = get_legal_moves(next_state, 1)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask,
        max_actions=next_state.board.width * next_state.board.height * 4,
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    next_state = move(next_state, 1, action[1], action[0], action[2])
    next_state = reinforce_troops(next_state, config)
    return next_state


@partial(jax.jit, static_argnames=("config",))
def p1_step(
    state: EnvState,
    rng_key: jnp.ndarray,
    config: EnvConfig,
    action: jnp.ndarray,
) -> tuple[EnvState, jnp.ndarray]:
    # p1 move
    next_state: EnvState = move(state, 0, action[1], action[0], action[2])
    # p2 move
    legal_actions_mask = get_legal_moves(next_state, 1)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask,
        max_actions=next_state.board.width * next_state.board.height * 4,
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    next_state = move(next_state, 1, action[1], action[0], action[2])
    next_state = reinforce_troops(next_state, config)
    reward_p1 = reward_function(state, next_state, 0)
    return next_state, reward_p1
