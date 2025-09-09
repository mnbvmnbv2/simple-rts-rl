from functools import partial

import jax
import jax.numpy as jnp

from src.rts.state import EnvState
from src.rts.config import EnvConfig
from src.rts.env import move, reinforce_troops, reward_function


@jax.jit
def get_legal_moves(state: EnvState, player: int) -> jnp.ndarray:
    board = state.board
    # Select the troop array for the given player.
    troop_array = board.player_troops[player]

    # A move is only legal if the source cell has more than 1 troop.
    active = troop_array > 1

    width = board.width
    height = board.height
    # Create a grid of coordinates.
    # Use 'xy' indexing so that the resulting arrays have shape (height, width)
    I, J = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="xy")

    # Compute boundary conditions for each direction.
    up_possible = (J > 0) & active
    right_possible = (I < width - 1) & active
    down_possible = (J < height - 1) & active
    left_possible = (I > 0) & active

    # Stack the directional legal move masks into the last dimension.
    legal_moves = jnp.stack(
        [up_possible, right_possible, down_possible, left_possible], axis=-1
    )
    return legal_moves


def fixed_argwhere(mask, max_actions: int = 100):
    # Here, we force the output of argwhere to have shape (max_actions, mask.ndim).
    indices = jnp.argwhere(mask, size=max_actions, fill_value=-1)
    # Compute the actual number of legal actions as a scalar.
    num_actions = jnp.sum(mask)
    return indices, num_actions


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
    reward_p1 = reward_function(state, next_state, 0, config.reward_config)
    return next_state, reward_p1
