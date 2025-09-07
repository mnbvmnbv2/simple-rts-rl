import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Board:
    player_troops: jnp.ndarray  # (player, width, height) of int
    neutral_troops: jnp.ndarray  # (width, height) of int
    bases: jnp.ndarray  # (width, height) of bool

    @property
    def num_players(self) -> int:
        return self.player_troops.shape[0]

    @property
    def width(self) -> int:
        return self.player_troops.shape[2]

    @property
    def height(self) -> int:
        return self.player_troops.shape[1]

    @jax.jit
    def flatten(self):
        # reductions to scalars
        max_player = jnp.max(self.player_troops)
        max_neutral = jnp.max(self.neutral_troops)

        # elementwise max of the two scalars
        max_troops = jnp.maximum(max_player, max_neutral)

        # avoid divide-by-zero if the board is empty
        denom = jnp.where(max_troops > 0, max_troops, 1)

        return jnp.concatenate(
            [
                (self.player_troops / denom).ravel().astype(jnp.float32),
                (self.neutral_troops / denom).ravel().astype(jnp.float32),
                self.bases.astype(jnp.float32).ravel(),
            ]
        )


@struct.dataclass
class EnvState:
    board: Board
    time: jnp.ndarray
