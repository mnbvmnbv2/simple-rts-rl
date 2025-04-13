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
        return jnp.concatenate(
            [
                self.player_troops.flatten(),
                self.neutral_troops.flatten(),
                self.bases.flatten(),
            ]
        )


@struct.dataclass
class EnvState:
    board: Board
    time: int = 5
