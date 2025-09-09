import chex
import jax.numpy as jnp

from src.rts.state import EnvState


def assert_valid_state(state: EnvState) -> None:
    board = state.board
    # Check types
    chex.assert_type(state.board.player_troops, jnp.integer)
    chex.assert_type(board.neutral_troops, jnp.integer)
    chex.assert_type(board.bases, jnp.bool)

    # Check that all values are non-negative.
    assert jnp.all(board.player_troops >= 0), "Negative player troops"
    assert jnp.all(board.neutral_troops >= 0), "Negative neutral troops"

    # Check that no tile has troops from multiple players (only one channel from 0 to 2 can be over 0).
    troop_presence = (
        (board.player_troops > 0).astype(jnp.int32)  # FIXME
        + (board.neutral_troops > 0).astype(jnp.int32)
    )
    assert jnp.all(troop_presence <= 1), "Some tiles have troops from multiple players."

    # Check time is not negative.
    assert state.time >= 0, "Time is negative."
