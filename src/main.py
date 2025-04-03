from functools import partial

import jax
import jax.numpy as jnp
from src.rts.config import EnvConfig
from src.rts.env import EnvState, init_state, move, reinforce_troops, reward_function
from src.rts.utils import get_legal_moves, fixed_argwhere
