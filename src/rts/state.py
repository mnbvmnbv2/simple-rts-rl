import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Board:
    player_1_troops: jnp.ndarray  # shape: (width, height) of int
    player_2_troops: jnp.ndarray  # shape: (width, height) of int
    neutral_troops: jnp.ndarray  # shape: (width, height) of int
    bases: jnp.ndarray  # shape: (width, height) of bool

    @property
    def width(self) -> int:
        return self.player_1_troops.shape[1]

    @property
    def height(self) -> int:
        return self.player_1_troops.shape[0]

    @jax.jit
    def flatten(self):
        return jnp.concatenate(
            [
                self.player_1_troops.flatten(),
                self.player_2_troops.flatten(),
                self.neutral_troops.flatten(),
                self.bases.flatten(),
            ]
        )


@struct.dataclass
class EnvState:
    board: Board
    time: int = 5
