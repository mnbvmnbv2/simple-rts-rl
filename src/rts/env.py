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

    num_special = (
        config.num_players + config.num_neutral_bases + config.num_neutral_troops_start
    )
    assert total_cells >= num_special, "Board too small for the required placements."

    rng_key, subkey = jax.random.split(rng_key)
    all_indices = jax.random.permutation(subkey, total_cells)

    cut_1 = config.num_players
    cut_2 = cut_1 + config.num_neutral_bases
    cut_3 = cut_2 + config.num_neutral_troops_start
    player_base_indices = all_indices[:cut_1]
    neutral_base_indices = all_indices[cut_1:cut_2]
    neutral_troop_indices = all_indices[cut_2:cut_3]

    player_troops_flat = jnp.zeros((config.num_players, total_cells), dtype=jnp.int32)
    neutral_troops_flat = jnp.zeros(total_cells, dtype=jnp.int32)
    bases_flat = jnp.zeros(total_cells, dtype=jnp.bool_)

    # Set player bases with troops.
    player_troops_flat = player_troops_flat.at[
        (jnp.arange(config.num_players), player_base_indices)
    ].set(config.player_start_troops)
    bases_flat = bases_flat.at[player_base_indices].set(True)

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
    player_troops = player_troops_flat.reshape(
        (config.num_players, config.board_height, config.board_width)
    )
    neutral_troops = neutral_troops_flat.reshape(
        (config.board_height, config.board_width)
    )
    bases = bases_flat.reshape((config.board_height, config.board_width))

    board = Board(
        player_troops=player_troops,
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
    num_players = board.num_players

    player_troops = board.player_troops[player]

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

    # Compute attacking troops
    num_attacking_troops = player_troops[y, x] - 1

    # Gather enemy and neutral information at target cell
    enemy_counts = board.player_troops[:, target_y, target_x]
    mask = jnp.not_equal(jnp.arange(num_players), player)
    sum_enemy = jnp.sum(enemy_counts * mask.astype(jnp.int32))
    neutral_count = board.neutral_troops[target_y, target_x]
    total_enemy = sum_enemy + neutral_count

    # Compute the damage distribution if the move is valid
    damage_ratio = jnp.where(total_enemy > 0, num_attacking_troops / total_enemy, 0)
    damage_enemies = jnp.where(mask, jnp.floor(enemy_counts * damage_ratio), 0)
    damage_neutral = jnp.where(
        total_enemy > 0, jnp.floor(neutral_count * damage_ratio), 0
    )

    new_enemy_counts = jnp.maximum(0, enemy_counts - damage_enemies)
    new_neutral = jnp.maximum(0, neutral_count - damage_neutral)

    total_damage = jnp.sum(damage_enemies) + damage_neutral
    remaining_attacking_troops = jnp.maximum(0, num_attacking_troops - total_damage)

    # Update target cell for moving player
    new_moving_player_count = (
        player_troops[target_y, target_x] + remaining_attacking_troops
    )

    # Create a new target cell vector: update the moving player's cell and keep others updated as per new_enemy_counts
    new_target_troops = jnp.where(
        jnp.arange(num_players) == player, new_moving_player_count, new_enemy_counts
    )

    # Update troops for the moving player source and target cell (if valid)
    player_troops = board.player_troops
    player_troops = player_troops.at[player, y, x].set(
        jnp.where(valid_move, 1, player_troops[player, y, x])
    )
    player_troops = player_troops.at[:, target_y, target_x].set(
        jnp.where(valid_move, new_target_troops, player_troops[:, target_y, target_x])
    )

    # Update neutral troops at target cell
    new_neutral_troops = board.neutral_troops.at[target_y, target_x].set(
        jnp.where(valid_move, new_neutral, board.neutral_troops[target_y, target_x])
    )

    new_board = board.replace(
        player_troops=player_troops,
        neutral_troops=new_neutral_troops,
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

    troop_locations = board.player_troops > 0
    base_bonus = board.bases.astype(jnp.int32)

    player_troops = (
        board.player_troops
        + bonus_troops * troop_locations.astype(jnp.int32)
        + base_bonus * troop_locations.astype(jnp.int32)
    )

    new_board = board.replace(player_troops=player_troops)

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
    player_troops_current = state.board.player_troops[player]
    player_troops_next = next_state.board.player_troops[player]

    opponent_troops_current = (
        jnp.sum(state.board.player_troops, axis=0) - player_troops_current
    )
    opponent_troops_next = (
        jnp.sum(next_state.board.player_troops, axis=0) - player_troops_next
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
    """Check if the game is finished, when only one player has non-zero troops on the board."""
    total_troops_per_player = jnp.sum(state.board.player_troops, axis=(1, 2))
    active_players = total_troops_per_player > 0
    num_active = jnp.sum(active_players)
    return num_active == 1


@jax.jit
def random_move(
    state: EnvState,
    player: int,
    rng_key: jnp.ndarray,
) -> tuple[int, int, int]:
    legal_actions_mask = get_legal_moves(state, player)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask, max_actions=state.board.width * state.board.height * 4
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    return action


def player_move(carry, player):
    state, rng_key = carry
    rng_key, subkey = jax.random.split(rng_key)
    action = random_move(state, player, subkey)
    next_state = move(state, player, action[1], action[0], action[2])
    return (next_state, rng_key), None


@partial(jax.jit, static_argnames=("config",))
def random_step(
    state: EnvState,
    rng_key: jnp.ndarray,
    config: EnvConfig,
) -> EnvState:
    (new_state, rng_key), _ = jax.lax.scan(
        player_move, (state, rng_key), jnp.arange(config.num_players)
    )

    # After all players have moved, apply reinforcement.
    new_state = reinforce_troops(new_state, config)
    return new_state


@partial(jax.jit, static_argnames=("config",))
def p1_step(
    state: EnvState,
    rng_key: jnp.ndarray,
    config: EnvConfig,
    action: jnp.ndarray,
) -> tuple[EnvState, jnp.ndarray]:
    new_state: EnvState = move(state, 0, action[1], action[0], action[2])
    (new_state, rng_key), _ = jax.lax.scan(
        player_move, (new_state, rng_key), jnp.arange(1, config.num_players)
    )

    next_state = reinforce_troops(new_state, config)
    reward_p1 = reward_function(state, next_state, 0)
    return next_state, reward_p1
