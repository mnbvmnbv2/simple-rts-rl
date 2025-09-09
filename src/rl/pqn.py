import functools
import gc
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from src.rl.model import MLP
from src.rts.config import EnvConfig
from src.rts.env import init_state, is_done
from src.rts.utils import get_legal_moves, p1_step, get_random_move_for_player


class TimerLog:
    def __init__(self):
        self.store = defaultdict(list)

    @contextmanager
    def record(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.store[name].append(time.perf_counter() - t0)


@dataclass(frozen=True)
class Params:
    num_iterations: int
    lr: float
    gamma: float
    q_lambda: float
    num_envs: int
    num_steps: int
    update_epochs: int
    num_minibatches: int
    epsilon: float


@functools.partial(nnx.jit, static_argnames=("config", "num_steps"))
def single_rollout(
    rng_key,
    config: EnvConfig,
    model: nnx.Module,
    num_steps: int,
    epsilon: float,
):
    state = init_state(rng_key, config)

    def policy_step(carry, _):
        state, done, rng_key, cum_reward = carry

        # if done we init new state
        rng_key, subkey = jax.random.split(rng_key)
        state = jax.lax.cond(
            done,
            lambda _: init_state(subkey, config),
            lambda _: state,
            operand=None,
        )

        flat_state = state.board.flatten()
        legal_mask = get_legal_moves(state, 0)

        logits = model(flat_state)
        # choose the action with the highest Q-value that is also legal
        q_net_action = jnp.argmax((logits + 1000) * legal_mask)
        # epsilon-greedy exploration
        explore_action = get_random_move_for_player(state, 0, rng_key)

        action = jax.lax.cond(
            jax.random.bernoulli(rng_key, epsilon),
            lambda _: explore_action,
            lambda _: q_net_action,
            operand=None,
        )
        action = jnp.asarray(action, dtype=jnp.int32)

        rng_key, subkey = jax.random.split(rng_key)
        next_state, p1_reward = p1_step(state, subkey, config, action)

        new_cum_reward = cum_reward + p1_reward

        done = is_done(next_state)

        y = (state.board.flatten(), action, p1_reward, done, next_state.board.flatten())

        return (next_state, done, rng_key, new_cum_reward), y

    (final_state, final_done, final_rng, cum_return), scan_out = jax.lax.scan(
        policy_step,
        (state, jnp.array(False), rng_key, jnp.array(0.0)),
        None,
        num_steps,
    )
    obs_buffer, actions_buffer, rewards_buffer, done_buffer, next_obs_buffer = scan_out
    return (
        obs_buffer,
        actions_buffer,
        rewards_buffer,
        done_buffer,
        next_obs_buffer,
        cum_return,
    )


@nnx.jit
def q_lambda_return(
    q_net: nnx.Module,
    rewards_buffer: jnp.ndarray,
    done_buffer: jnp.ndarray,
    next_obs_buffer: jnp.ndarray,
    gamma: float,
    q_lambda: float,
) -> jnp.ndarray:
    # Compute Q-values for the next observations via vectorized max.
    # This returns an array of shape (num_steps,)
    q_values = jax.vmap(lambda obs: jnp.max(q_net(obs), axis=-1))(next_obs_buffer)

    # For the final step, compute the return as:
    # returns[-1] = rewards[-1] + gamma * q_value[-1] * (1 - done[-1])
    returns_last = rewards_buffer[-1] + gamma * q_values[-1] * (1.0 - done_buffer[-1])

    # For timesteps 0,...,num_steps-2 we use:
    # returns[t] = rewards[t] + gamma * (q_lambda * returns[t+1] +
    #                                    (1 - q_lambda) * q_value[t+1] * (1 - done[t+1]))
    # To compute this in reverse, we reverse the arrays (excluding the last step).
    rewards_rev = rewards_buffer[:-1][::-1]
    dones_rev = done_buffer[1:][::-1]
    next_vals_rev = q_values[1:][::-1]

    def scan_fn(next_return, inputs):
        reward, done, next_value = inputs
        nextnonterminal = 1.0 - done
        current_return = reward + gamma * (
            q_lambda * next_return + (1 - q_lambda) * next_value * nextnonterminal
        )
        return current_return, current_return

    # The scan will traverse the reversed sequences.
    # Its initial carry is the last return (for t = num_steps - 1)
    _, returns_rev_scan = jax.lax.scan(
        scan_fn, returns_last, (rewards_rev, dones_rev, next_vals_rev)
    )

    # Flip the scanned returns back to the original order.
    returns_first_part = returns_rev_scan[::-1]
    # Append the final return computed above.
    full_returns = jnp.concatenate(
        [returns_first_part, jnp.array([returns_last])], axis=0
    )

    return full_returns


@nnx.jit
def train_step(
    q_net: nnx.Module,
    optimizer: nnx.Optimizer,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    returns: jnp.ndarray,
) -> tuple[nnx.Module, nnx.Optimizer, jnp.ndarray]:
    def loss_fn(m) -> jnp.ndarray:
        q_values = m(observations)
        acted_q = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
        return jnp.mean((acted_q - returns) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(q_net)
    optimizer.update(grads)
    return q_net, optimizer, loss


vmapped_rollout = jax.vmap(single_rollout, in_axes=(0, None, None, None, None))
vmapped_q_lambda_return = jax.vmap(q_lambda_return, in_axes=(None, 0, 0, 0, None, None))


def train_minibatched(
    q_net: nnx.Module,
    optimizer: nnx.Optimizer,
    config: EnvConfig,
    params: Params,
    seed: int = 0,
):
    rng_key = jax.random.PRNGKey(seed)
    losses = []
    cum_returns = []
    timer = TimerLog()

    for iteration in tqdm(range(params.num_iterations)):
        with timer.record("rng_split"):
            rng_keys = jax.random.split(rng_key, params.num_envs + 1)
            rng_key, rollout_keys = rng_keys[0], rng_keys[1:]

        with timer.record("rollout"):
            rollout = vmapped_rollout(
                rollout_keys, config, q_net, params.num_steps, params.epsilon
            )
            (
                obs_buffer,
                actions_buffer,
                rewards_buffer,
                done_buffer,
                next_obs_buffer,
                cum_return,
            ) = rollout

        cum_return = jax.device_get(cum_return)
        cum_returns.append(np.asarray(cum_return))

        with timer.record("q_lambda_return"):
            returns = vmapped_q_lambda_return(
                q_net,
                rewards_buffer,
                done_buffer,
                next_obs_buffer,
                params.gamma,
                params.q_lambda,
            )

        with timer.record("reshape"):
            flat_observations = obs_buffer.reshape(-1, obs_buffer.shape[-1])
            flat_actions = actions_buffer.reshape(-1)
            flat_returns = returns.reshape(-1)
            num_samples = flat_observations.shape[0]
            minibatch_size = num_samples // params.num_minibatches

        with timer.record("update"):
            for epoch in range(params.update_epochs):
                rng_key, perm_key = jax.random.split(rng_key)
                permuted_indices = jax.random.permutation(perm_key, num_samples)

                for i in range(params.num_minibatches):
                    start_idx = i * minibatch_size
                    # We might lose out on some final samples, but we do so to avoid recompile of train with new
                    # shape
                    end_idx = (i + 1) * minibatch_size
                    minibatch_idx = permuted_indices[start_idx:end_idx]

                    minibatch_obs = flat_observations[minibatch_idx]
                    minibatch_actions = flat_actions[minibatch_idx]
                    minibatch_returns = flat_returns[minibatch_idx]

                    _, _, loss = train_step(
                        q_net,
                        optimizer,
                        minibatch_obs,
                        minibatch_actions,
                        minibatch_returns,
                    )
                    loss = loss.block_until_ready()
                    losses.append(float(loss))

    del (
        obs_buffer,
        actions_buffer,
        rewards_buffer,
        done_buffer,
        next_obs_buffer,
        returns,
    )
    del flat_observations, flat_actions, flat_returns
    gc.collect()
    jax.clear_caches()

    times_dict = dict(timer.store)
    compile_time = sum(v[0] for k, v in times_dict.items())
    print(f"{compile_time=}")

    return q_net, losses, cum_returns, times_dict


if __name__ == "__main__":
    width = 10
    height = 10
    config = EnvConfig(
        num_players=2,
        board_width=width,
        board_height=height,
        num_neutral_bases=3,
        num_neutral_troops_start=5,
        neutral_troops_min=4,
        neutral_troops_max=10,
        player_start_troops=5,
        bonus_time=10,
    )
    params = Params(
        num_iterations=500,
        lr=4e-4,
        gamma=0.99,
        q_lambda=0.92,
        num_envs=50,
        num_steps=250,
        update_epochs=2,
        num_minibatches=4,
        epsilon=0.3,
    )
    q_net = MLP(width * height * 4, [256, 256], width * height * 4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(params.lr))
    train_minibatched(q_net, optimizer, config, params)
