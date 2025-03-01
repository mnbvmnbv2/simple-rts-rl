import jax
import jax.numpy as jnp
from src.rts.config import EnvConfig
from src.rts.env import EnvState, init_state, move, increase_troops
from src.rts.utils import get_legal_moves, fixed_argwhere


@jax.jit
def step(state: EnvState, rng_key):
    # p1 move
    legal_actions_mask = get_legal_moves(state, 0)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask, max_actions=state.board.width * state.board.height * 4
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    state = move(state, 0, action[1], action[0], action[2])
    # p2 move
    legal_actions_mask = get_legal_moves(state, 1)
    legal_actions, num_actions = fixed_argwhere(
        legal_actions_mask, max_actions=state.board.width * state.board.height * 4
    )
    rng_key, subkey = jax.random.split(rng_key)
    action_idx = jax.random.randint(subkey, (), 0, num_actions)
    action = jnp.take(legal_actions, action_idx, axis=0)
    state = move(state, 1, action[1], action[0], action[2])
    state = increase_troops(state)
    return state


def main():
    config = EnvConfig(
        board_width=10,
        board_height=10,
        num_neutral_bases=4,
        num_neutral_troops_start=8,
        neutral_bases_min_troops=1,
        neutral_bases_max_troops=10,
    )
    state = init_state(jax.random.PRNGKey(0), config)
    rng_key = jax.random.PRNGKey(0)
    step(state, rng_key)


if __name__ == "__main__":
    main()
