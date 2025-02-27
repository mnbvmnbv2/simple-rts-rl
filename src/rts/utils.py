import chex
import jax.numpy as jnp

from src.rts.env import EnvState


def get_legal_moves(state: EnvState, player: int) -> jnp.ndarray:
    board = state.board
    legal_moves = jnp.zeros((board.shape[0], board.shape[1], 4), dtype=jnp.bool_)
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j, player] > 1:
                legal_moves = legal_moves.at[i, j, 0].set(i > 0)
                legal_moves = legal_moves.at[i, j, 1].set(j < board.shape[1] - 1)
                legal_moves = legal_moves.at[i, j, 2].set(i < board.shape[0] - 1)
                legal_moves = legal_moves.at[i, j, 3].set(j > 0)
    return legal_moves


def assert_valid_state(state: EnvState) -> None:
    # Check that the board is of the right shape
    chex.assert_shape(state.board, (10, 10, 4))
    # Check that the number of troops and bases are integers
    chex.assert_type(state.board, jnp.integer)
    # Check that all values are non-negative.
    assert jnp.all(state.board >= 0), "Board has negative values."

    # For tiles that are bases, ensure at least one troop.
    base_valid = jnp.where(
        state.board[..., 3] == 1, jnp.sum(state.board[..., :3]) > 0, True
    )
    assert jnp.all(base_valid), "Some bases do not have any troops."

    # Check that no tile has multiple bases (channel 3 at most 1).
    no_multiple_bases = state.board[..., 3] <= 1
    assert jnp.all(no_multiple_bases), "Some tiles have multiple bases."

    # Check that no tile has troops from multiple players (only one channel from 0 to 2 can be over 0).
    no_multiple_troops = jnp.sum(state.board[..., :3] > 0, axis=-1) <= 1
    assert jnp.all(no_multiple_troops), "Some tiles have troops from multiple players."
