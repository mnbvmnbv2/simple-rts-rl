from functools import partial

import jax
import jax.numpy as jnp

from src.rts.config import EnvConfig
from src.rts.env import move, reinforce_troops, reward_function
from src.rts.state import EnvState


def encode_action(y: int, x: int, d: int, width: int) -> int:
    return y * (width * 4) + x * 4 + d


def decode_action(a: int, width: int):
    y = a // (width * 4)
    x = (a % (width * 4)) // 4
    d = a % 4
    return y, x, d


@jax.jit
def get_legal_moves(state: EnvState, player: int) -> jnp.ndarray:
    board = state.board
    player_troop_array = board.player_troops[player]

    # A move is only legal if the source cell has more than 1 troop.
    active = player_troop_array > 1

    width, height = board.width, board.height
    # Create a grid of coordinates.
    # Use 'xy' indexing so that the resulting arrays have shape (height, width)
    I, J = jnp.meshgrid(jnp.arange(width), jnp.arange(height), indexing="xy")

    # Compute boundary conditions for each direction.
    up = (J > 0) & active
    right = (I < width - 1) & active
    down = (J < height - 1) & active
    left = (I > 0) & active

    # Stack the directional legal move masks into the last dimension.
    legal_moves = jnp.stack([up, right, down, left], axis=-1)
    return legal_moves.ravel()


@jax.jit
def sample_legal_action_flat(rng_key: jnp.ndarray, legal_mask_flat: jnp.ndarray) -> int:
    # Convert bool mask to probabilities
    logits = jnp.where(legal_mask_flat, 0.0, -1e9)  # big negative = impossible
    # Gumbel-max trick is numerically stable & JIT friendly
    g = jax.random.gumbel(rng_key, logits.shape)
    return jnp.argmax(logits + g)


@jax.jit
def move_by_action_int(state: EnvState, player: int, action_int: int) -> EnvState:
    W = state.board.width
    y, x, d = decode_action(action_int, W)
    return move(state, player, x, y, d)


def get_random_move_for_player(
    state: EnvState, player: int, rng_key: jnp.ndarray
) -> int:
    legal_actions_mask = get_legal_moves(state, player)
    return sample_legal_action_flat(rng_key, legal_actions_mask)


def do_random_move_for_player(carry, player: int):
    state, rng_key = carry
    rng_key, subkey = jax.random.split(rng_key)
    action_int = get_random_move_for_player(state, player, subkey)
    next_state = move_by_action_int(state, player, action_int)
    return (next_state, rng_key), None


@partial(jax.jit, static_argnames=("config",))
def p1_step(
    state: EnvState,
    rng_key: jnp.ndarray,
    config: EnvConfig,
    action: jnp.ndarray,
) -> tuple[EnvState, jnp.ndarray]:
    new_state: EnvState = move_by_action_int(state, 0, action)
    (new_state, rng_key), _ = jax.lax.scan(
        do_random_move_for_player,
        (new_state, rng_key),
        jnp.arange(1, config.num_players),
    )

    next_state = reinforce_troops(new_state, config)
    reward_p1 = reward_function(state, next_state, 0, config.reward_config)
    return next_state, reward_p1
