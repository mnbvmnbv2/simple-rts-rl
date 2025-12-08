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
from src.rts.env import EnvState, init_state, is_done
from src.rts.utils import get_legal_moves, p1_step, sample_legal_action_flat
from src.rts.state import get_cnn_observation


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
    """
    One ε-greedy rollout for player 0.
    Returns:
      obs_buffer         (T, obs_dim)
      actions_buffer     (T,)
      rewards_buffer     (T,)
      done_buffer        (T,)
      next_obs_buffer    (T, obs_dim)
      next_legal_buffer  (T, num_actions)
      cum_return         ()
    """
    state = init_state(rng_key, config)

    def policy_step(carry, _):
        state, done, rng_key, cum_reward = carry

        rng_key, k_reset, k_eps, k_explore, k_step = jax.random.split(rng_key, 5)

        # if done we init new state
        state: EnvState = jax.lax.cond(
            done, lambda _: init_state(k_reset, config), lambda _: state, operand=None
        )

        # Current obs + legal mask
        # obs = state.board.flatten()
        obs = get_cnn_observation(state, 0)
        legal_mask = get_legal_moves(state, 0)  # (num_actions,)

        # choose the action with the highest Q-value that is also legal
        q_vals = model(obs)  # (num_actions,)
        neg_inf = jnp.array(-jnp.inf, dtype=q_vals.dtype)
        q_masked = jnp.where(legal_mask, q_vals, neg_inf)
        exploit_action = jnp.argmax(q_masked)

        # epsilon-greedy exploration
        explore_action = sample_legal_action_flat(k_explore, legal_mask)
        action = jnp.where(
            jax.random.bernoulli(k_eps, epsilon), explore_action, exploit_action
        )
        action = action.astype(jnp.int32)

        # env transition
        next_state, reward = p1_step(state, k_step, config, action)

        done_next = is_done(next_state)
        # next_obs = next_state.board.flatten()
        next_obs = get_cnn_observation(next_state, 0)
        next_legal_mask = get_legal_moves(next_state, 0)

        new_cum_reward = (cum_reward + reward).astype(jnp.float32)
        reward = reward.astype(jnp.float32)

        y = (obs, action, reward, done_next, next_obs, next_legal_mask)
        return (next_state, done_next, rng_key, new_cum_reward), y

    (final_state, final_done, final_rng, cum_return), scan_out = jax.lax.scan(
        policy_step,
        (state, jnp.array(False), rng_key, jnp.array(0.0, dtype=jnp.float32)),
        None,
        num_steps,
    )

    (
        obs_buffer,
        actions_buffer,
        rewards_buffer,
        done_buffer,
        next_obs_buffer,
        next_legal_buffer,
    ) = scan_out

    return (
        obs_buffer,
        actions_buffer,
        rewards_buffer,
        done_buffer,
        next_obs_buffer,
        next_legal_buffer,
        cum_return,
    )


@nnx.jit
def q_lambda_return(
    q_net: nnx.Module,
    rewards_buffer: jnp.ndarray,
    done_buffer: jnp.ndarray,
    next_obs_buffer: jnp.ndarray,
    next_legal_buffer: jnp.ndarray,
    gamma: float,
    q_lambda: float,
) -> jnp.ndarray:
    """
    Q(λ) targets with masked bootstrapping and hard guards against NaNs.

    Recurrence with terminal step T-1:
      V_t      = max_a Q(next_obs_t, a) over legal a, else 0
      V_t      = 0 if done_t
      R_{T-1}  = r_{T-1} + gamma * V_{T-1}
      R_t      = r_t + gamma * ( q_lambda * R_{t+1} + (1 - q_lambda) * V_t )
    """
    rewards = rewards_buffer.astype(jnp.float32)  # (T,)
    done_f = done_buffer.astype(jnp.float32)  # (T,)
    not_done = 1.0 - done_f  # (T,)

    def masked_qmax(ob, legal_mask):
        q = q_net(ob)  # (A,)
        # Ensure same dtype for filler
        neg_inf = jnp.array(-jnp.inf, dtype=q.dtype)
        q_masked = jnp.where(legal_mask, q, neg_inf)  # illegal → -inf
        qmax = jnp.max(q_masked, axis=-1)  # may be -inf if none legal
        any_legal = jnp.any(legal_mask)
        # If no legal actions, define value 0.0 to avoid -inf
        qmax = jnp.where(any_legal, qmax, jnp.array(0.0, dtype=q.dtype))
        return qmax.astype(jnp.float32)

    # Compute bootstrap values and drop them to 0 when done
    v_next = jax.vmap(masked_qmax)(next_obs_buffer, next_legal_buffer)  # (T,)
    v_next = jnp.where(not_done > 0.0, v_next, 0.0)  # (T,)

    # Terminal step
    last = rewards[-1] + gamma * v_next[-1]  # both float32, safe

    # Reverse-scan for t = T-2..0
    rew_rev = rewards[:-1][::-1]
    v_rev = v_next[:-1][::-1]

    def scan_fn(R_next, inputs):
        r_t, v_t = inputs
        boot = q_lambda * R_next + (1.0 - q_lambda) * v_t
        R_t = r_t + gamma * boot
        return R_t, R_t

    _, R_rev = jax.lax.scan(scan_fn, last, (rew_rev, v_rev))
    R_fwd = R_rev[::-1]
    targets = jnp.concatenate([R_fwd, jnp.array([last], dtype=jnp.float32)], axis=0)
    # Final safety: replace any residual NaNs/Infs (shouldn't happen with guards above)
    targets = jnp.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    return targets  # (T,)


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
vmapped_q_lambda_return = jax.vmap(
    q_lambda_return, in_axes=(None, 0, 0, 0, 0, None, None)
)


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
                next_legal_buffer,
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
                next_legal_buffer,
                params.gamma,
                params.q_lambda,
            )

        with timer.record("reshape"):
            # flat_observations = obs_buffer.reshape(-1, obs_buffer.shape[-1])
            flat_observations = obs_buffer.reshape(-1, *obs_buffer.shape[2:])
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
