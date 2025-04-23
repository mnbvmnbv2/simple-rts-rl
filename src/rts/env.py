from functools import partial

import jax
import jax.numpy as jnp

from src.rts.config import EnvConfig, RewardConfig
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

    # Check move validity.
    source_in_bounds = jnp.logical_and(x >= 0, x < board.width) & jnp.logical_and(
        y >= 0, y < board.height
    )
    target_in_bounds = jnp.logical_and(
        target_y >= 0, target_y < board.height
    ) & jnp.logical_and(target_x >= 0, target_x < board.width)
    has_enough_troops = player_troops[y, x] > 1
    valid_move = source_in_bounds & target_in_bounds & has_enough_troops

    # Number of attacking troops leaving the source (keep one behind).
    num_attacking_troops = player_troops[y, x] - 1

    enemy_counts = board.player_troops[:, target_y, target_x]
    mask = jnp.not_equal(jnp.arange(num_players), player)
    sum_enemy = jnp.sum(enemy_counts * mask.astype(jnp.int32))
    neutral_count = board.neutral_troops[target_y, target_x]
    total_enemy = sum_enemy + neutral_count

    damage = jnp.minimum(num_attacking_troops, total_enemy)
    surviving_attackers = num_attacking_troops - damage

    # Reduce enemy troops proportionally.
    new_enemy_counts = jnp.where(
        total_enemy > 0,
        enemy_counts - (enemy_counts / total_enemy) * damage,
        enemy_counts,
    )
    new_enemy_counts = jnp.floor(new_enemy_counts)  # convert to integer counts

    new_neutral = jnp.where(
        total_enemy > 0,
        neutral_count - (neutral_count / total_enemy) * damage,
        neutral_count,
    )
    new_neutral = jnp.floor(new_neutral)

    # Update the moving player's count at the target cell.
    new_moving_player_count = (
        board.player_troops[player, target_y, target_x] + surviving_attackers
    )

    # Construct new target cell vector: for the moving player, use new_moving_player_count;
    # for all other players, use the updated enemy counts.
    new_target_troops = jnp.where(
        jnp.arange(num_players) == player,
        new_moving_player_count,
        new_enemy_counts,
    )

    # Update the board arrays (only if the move is valid).
    updated_player_troops = board.player_troops
    updated_player_troops = updated_player_troops.at[player, y, x].set(
        jnp.where(valid_move, 1, board.player_troops[player, y, x])
    )
    updated_player_troops = updated_player_troops.at[:, target_y, target_x].set(
        jnp.where(
            valid_move, new_target_troops, board.player_troops[:, target_y, target_x]
        )
    )
    updated_neutral_troops = board.neutral_troops.at[target_y, target_x].set(
        jnp.where(valid_move, new_neutral, board.neutral_troops[target_y, target_x])
    )

    new_board = board.replace(
        player_troops=updated_player_troops,
        neutral_troops=updated_neutral_troops,
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


def _tile_changes(mask_cur, mask_nxt):
    cur = jnp.sum(mask_cur)
    nxt = jnp.sum(mask_nxt)
    gain = jnp.maximum(nxt - cur, 0)
    loss = jnp.maximum(cur - nxt, 0)
    return gain, loss


def _base_changes(player_mask_cur, player_mask_nxt, base_mask):
    return _tile_changes(player_mask_cur & base_mask, player_mask_nxt & base_mask)


def _neutral_changes(neu_cur, neu_nxt, player_cur, player_nxt):
    gain = jnp.sum(neu_cur & (~neu_nxt) & player_nxt)  # neutral → player
    loss = jnp.sum((~neu_cur) & neu_nxt & player_cur)  # player  → neutral
    return gain, loss


@partial(jax.jit, static_argnames=("config",))
def reward_function(
    state, next_state, player: int, config: RewardConfig
) -> jnp.ndarray:
    """Per-step reward for `player`, parameterised by `config`."""

    # ------------ Layers -----------------------------------------------------
    p_cur = state.board.player_troops[player]
    p_nxt = next_state.board.player_troops[player]

    opp_cur = jnp.sum(state.board.player_troops, axis=0) - p_cur
    opp_nxt = jnp.sum(next_state.board.player_troops, axis=0) - p_nxt

    base_mask = state.board.bases  # (h, w) bool

    n_cur = state.board.neutral_troops
    n_nxt = next_state.board.neutral_troops
    neu_cur = n_cur > 0
    neu_nxt = n_nxt > 0

    # ------------ Changes ----------------------------------------------------
    tile_gain, tile_loss = _tile_changes(p_cur > 0, p_nxt > 0)
    base_gain, base_loss = _base_changes(p_cur > 0, p_nxt > 0, base_mask)
    neut_gain, neut_loss = _neutral_changes(neu_cur, neu_nxt, p_cur > 0, p_nxt > 0)
    opp_tgain, opp_tloss = _tile_changes(opp_cur > 0, opp_nxt > 0)
    opp_bgain, opp_bloss = _base_changes(opp_cur > 0, opp_nxt > 0, base_mask)

    # ------------ Victory / defeat ------------------------------------------
    opp_tiles_cur = jnp.sum(opp_cur > 0)
    opp_tiles_nxt = jnp.sum(opp_nxt > 0)

    victory = ((opp_tiles_cur > 0) & (opp_tiles_nxt == 0)).astype(jnp.int32)
    defeat = ((tile_gain + tile_loss > 0) & (jnp.sum(p_nxt > 0) == 0)).astype(jnp.int32)

    # ------------ Weighted sum ----------------------------------------------
    reward = (
        # player tiles / bases
        tile_gain * config.tile_gain_reward
        + tile_loss * config.tile_loss_reward
        + base_gain * config.base_gain_reward
        + base_loss * config.base_loss_reward
        # neutral conversions
        + neut_gain * config.neutral_tile_gain_reward
        # opponent territory
        + opp_tloss * config.opponent_tile_loss_reward
        + opp_tgain * config.opponent_tile_gain_reward
        # opponent bases  (NEW)
        + opp_bloss * config.opponent_base_loss_reward
        + opp_bgain * config.opponent_base_gain_reward
        # end-of-game
        + victory * config.victory_reward
        + defeat * config.defeat_reward
    ).astype(jnp.float32)

    return reward


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
