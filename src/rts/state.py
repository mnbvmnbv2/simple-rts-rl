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


def get_cnn_observation(state: EnvState, player_id: int) -> jnp.ndarray:
    """
    Creates a (Height, Width, 6) tensor for the CNN.
    Normalizes troop counts using log1p (log(1+x)).
    Aligns the 'My Troops' channel to always be the requesting player_id.
    """
    board = state.board

    # --- 1. Troops Channels ---
    # Rotate player stack so 'player_id' is at index 0 (Relative Perspective)
    # shape: (num_players, H, W)
    rotated_players = jnp.roll(board.player_troops, -player_id, axis=0)

    # Channel 0: "My" troops
    my_troops = rotated_players[0]

    # Channel 1: "Enemy" troops (sum of all others for 2+ players)
    # If strictly 2 players, this is just rotated_players[1]
    enemy_troops = jnp.sum(rotated_players[1:], axis=0)

    # Channel 2: Neutral troops
    neutral_troops = board.neutral_troops

    # Normalize counts (Logarithmic scaling is crucial for RTS)
    # Using log(1 + x) prevents 0-troop tiles from exploding and
    # squashes massive stacks (e.g. 50 vs 100) into a readable range.
    c1 = jnp.log1p(my_troops)
    c2 = jnp.log1p(enemy_troops)
    c3 = jnp.log1p(neutral_troops)

    # --- 2. Base Channels ---
    # Bases is just a boolean mask of WHERE bases are.
    # We combine it with troop locations to find ownership.
    bases_mask = board.bases

    # Channel 4: My Bases
    c4 = bases_mask & (my_troops > 0)

    # Channel 5: Enemy Bases
    c5 = bases_mask & (enemy_troops > 0)

    # Channel 6: Neutral/Empty Bases
    # (A base is neutral if it has neutral troops OR no troops)
    c6 = bases_mask & (c4 == 0) & (c5 == 0)

    # --- 3. Stack ---
    # Stack along the last dimension to get (H, W, 6)
    obs = jnp.stack([c1, c2, c3, c4, c5, c6], axis=-1)

    return obs.astype(jnp.float32)
