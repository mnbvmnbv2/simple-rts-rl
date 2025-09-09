import jax
import jax.numpy as jnp
from flax import nnx
from jax import lax

from src.rts.config import EnvConfig
from src.rts.env import init_state
from src.rts.utils import get_legal_moves, p1_step


def evaluate_batch(
    q_net: nnx.Module,
    config: EnvConfig,
    rng_key: jax.random.PRNGKey,
    batch_size: int = 32,
    num_steps: int = 250,
) -> jnp.ndarray:
    init_keys = jax.random.split(rng_key, batch_size)
    states = jax.vmap(lambda key: init_state(key, config))(init_keys)

    def step_fn(states, step_key):
        legal_mask = jax.vmap(lambda s: get_legal_moves(s, 0))(states)
        boards_flat = jax.vmap(lambda s: s.board.flatten())(states)
        q_vals = jax.vmap(q_net)(boards_flat)
        masked_q = (q_vals + 1000) * legal_mask
        actions = jnp.argmax(masked_q, axis=1)

        subkeys = jax.random.split(step_key, batch_size)
        new_states, rewards = jax.vmap(lambda s, k, a: p1_step(s, k, config, a))(
            states, subkeys, actions
        )

        return new_states, rewards

    scan_keys = jax.random.split(rng_key, num_steps)

    def scan_func(carry, key):
        s, _ = carry
        new_s, r = step_fn(s, key)
        return (new_s, None), r

    (_, _), rewards_stack = lax.scan(
        scan_func,
        (states, None),
        scan_keys,
    )
    returns = jnp.sum(rewards_stack, axis=0)

    return returns
