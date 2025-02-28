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

    # For tiles that are bases, ensure at least one troop.
    total_troops = board.player_1_troops + board.player_2_troops + board.neutral_troops
    assert jnp.all(
        jnp.where(board.bases, total_troops > 0, True)
    ), "Some bases do not have any troops."

    # Check that no tile has troops from multiple players (only one channel from 0 to 2 can be over 0).
    troop_presence = (
        (board.player_1_troops > 0).astype(jnp.int32)
        + (board.player_2_troops > 0).astype(jnp.int32)
        + (board.neutral_troops > 0).astype(jnp.int32)
    )
    assert jnp.all(troop_presence <= 1), "Some tiles have troops from multiple players."

    # Check time is not negative.
    assert state.time >= 0, "Time is negative."
