import chex
import jax
import jax.numpy as jnp

from src.rts.env import EnvState


def assert_valid_state(state: EnvState) -> None:
    board = state.board
    # Check types
    chex.assert_type(state.board.player_1_troops, jnp.integer)
    chex.assert_type(board.player_2_troops, jnp.integer)
    chex.assert_type(board.neutral_troops, jnp.integer)
    chex.assert_type(board.bases, jnp.bool)

    # Check that all values are non-negative.
    assert jnp.all(board.player_1_troops >= 0), "Negative player 1 troops"
    assert jnp.all(board.player_2_troops >= 0), "Negative player 2 troops"
    assert jnp.all(board.neutral_troops >= 0), "Negative neutral troops"

    # Check that no tile has troops from multiple players (only one channel from 0 to 2 can be over 0).
    troop_presence = (
        (board.player_1_troops > 0).astype(jnp.int32)
        + (board.player_2_troops > 0).astype(jnp.int32)
        + (board.neutral_troops > 0).astype(jnp.int32)
    )
    assert jnp.all(troop_presence <= 1), "Some tiles have troops from multiple players."

    # Check time is not negative.
    assert state.time >= 0, "Time is negative."


@jax.jit
def get_legal_moves(state: EnvState, player: int) -> jnp.ndarray:
    board = state.board
    # Select the troop array for the given player.
    troop_array = jnp.where(player == 0, board.player_1_troops, board.player_2_troops)

    # A move is only legal if the source cell has more than 1 troop.
    active = troop_array > 1

    width = board.width
    height = board.height
    # Create a grid of coordinates.
    x_coords = jnp.arange(width)
    y_coords = jnp.arange(height)
    # Use meshgrid with indexing 'ij' so that I corresponds to x and J to y.
    I, J = jnp.meshgrid(x_coords, y_coords, indexing="ij")

    # Compute boundary conditions for each direction.
    # Direction 0: move left -> legal if x > 0.
    left_possible = (I > 0) & active
    # Direction 1: move down -> legal if y < height - 1.
    down_possible = (J < height - 1) & active
    # Direction 2: move right -> legal if x < width - 1.
    right_possible = (I < width - 1) & active
    # Direction 3: move up -> legal if y > 0.
    up_possible = (J > 0) & active

    # Stack the directional legal move masks into the last dimension.
    legal_moves = jnp.stack(
        [left_possible, down_possible, right_possible, up_possible], axis=-1
    )
    return legal_moves


def fixed_argwhere(mask, max_actions: int = 100):
    # Here, we force the output of argwhere to have shape (max_actions, mask.ndim).
    indices = jnp.argwhere(mask, size=max_actions, fill_value=-1)
    # Compute the actual number of legal actions as a scalar.
    num_actions = jnp.sum(mask)
    return indices, num_actions
