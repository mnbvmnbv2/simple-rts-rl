import functools
from dataclasses import dataclass
import time
from typing import NamedTuple

from flax import nnx
import optax
import jax
import jax.lax
import jax.numpy as jnp
from tqdm import tqdm

from src.rts.config import EnvConfig
from src.rts.env import init_state, is_done
from src.rts.utils import get_legal_moves, p1_step, random_move


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


class Model(nnx.Module):
    def __init__(self, in_dim, mid_dim, out_dim, rngs: nnx.Rngs):
        self.lin_in = nnx.Linear(in_dim, mid_dim, rngs=rngs)
        self.layer_norm1 = nnx.LayerNorm(mid_dim, rngs=rngs)
        self.lin_mid1 = nnx.Linear(mid_dim, mid_dim, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(mid_dim, rngs=rngs)
        self.lin_mid2 = nnx.Linear(mid_dim, mid_dim, rngs=rngs)
        self.layer_norm3 = nnx.LayerNorm(mid_dim, rngs=rngs)
        self.lin_out = nnx.Linear(mid_dim, out_dim, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer_norm1(self.lin_in(x)))
        x = nnx.relu(self.layer_norm2(self.lin_mid1(x)))
        x = nnx.relu(self.layer_norm3(self.lin_mid2(x)))
        return self.lin_out(x)


@functools.partial(nnx.jit, static_argnames=("config", "num_steps"))
def single_rollout(
    rng_key,
    config: EnvConfig,
    model: Model,
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
        legal_mask = get_legal_moves(state, 0).flatten()

        logits = model(flat_state)
        # choose the action with the highest Q-value that is also legal
        q_net_action = jnp.argmax((logits + 1000) * legal_mask)
        # epsilon-greedy exploration
        explore_action = random_move(state, 0, rng_key)[0]

        action = jax.lax.cond(
            jax.random.bernoulli(rng_key, epsilon),
            lambda _: explore_action,
            lambda _: q_net_action,
            operand=None,
        )

        # Split the scalar action into (row, col, direction) components.
        action_split = jnp.array(
            [
                action // (config.board_width * 4),
                (action % (config.board_width * 4)) // 4,
                action % 4,
            ]
        )

        rng_key, subkey = jax.random.split(rng_key)
        next_state, p1_reward = p1_step(state, subkey, config, action_split)

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
    q_net: Model,
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
    q_net: Model,
    optimizer: nnx.Optimizer,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    returns: jnp.ndarray,
) -> tuple[Model, nnx.Optimizer, jnp.ndarray]:
    def loss_fn(m) -> jnp.ndarray:
        q_values = m(observations)
        acted_q = jnp.take_along_axis(q_values, actions[:, None], axis=1).squeeze(-1)
        return jnp.mean((acted_q - returns) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(q_net)
    optimizer.update(grads)
    return q_net, optimizer, loss


vmapped_rollout = jax.vmap(single_rollout, in_axes=(0, None, None, None, None))
vmapped_q_lambda_return = jax.vmap(q_lambda_return, in_axes=(None, 0, 0, 0, None, None))


class TrainState(NamedTuple):
    q_net: Model
    opt: nnx.Optimizer
    rng: jax.Array


@functools.partial(nnx.jit, static_argnums=(1, 2))
def _iteration_step(state: TrainState, config: EnvConfig, params: Params):
    q_net, opt, rng = state

    # --- rollout ---
    rng, k_split = jax.random.split(rng)
    keys = jax.random.split(k_split, params.num_envs)
    obs_buf, act_buf, rew_buf, done_buf, next_obs_buf, cum_ret = vmapped_rollout(
        keys, config, q_net, params.num_steps, params.epsilon
    )
    returns = vmapped_q_lambda_return(
        q_net, rew_buf, done_buf, next_obs_buf, params.gamma, params.q_lambda
    )

    # --- flatten (N = num_envs * num_steps) ---
    flat_obs = obs_buf.reshape(-1, obs_buf.shape[-1])  # [N, D]
    flat_act = act_buf.reshape(-1)  # [N]
    flat_ret = returns.reshape(-1)  # [N]

    # --- static minibatch partition ---
    num_samples = flat_obs.shape[0]
    B = num_samples // params.num_minibatches
    M = params.num_minibatches
    num_full = M * B  # truncate remainder to keep shapes static

    flat_obs = flat_obs[:num_full]
    flat_act = flat_act[:num_full]
    flat_ret = flat_ret[:num_full]

    # ----- inner scan over minibatches -----
    @nnx.scan
    def scan_minibatches(carry_mb, xs_mb):
        q_net, opt = carry_mb
        mb_obs, mb_act, mb_ret = xs_mb  # each has leading axis B
        q_net, opt, loss = train_step(q_net, opt, mb_obs, mb_act, mb_ret)
        return (q_net, opt), loss

    # ----- outer scan over epochs -----
    @nnx.scan
    def scan_epochs(carry, _):
        q_net, opt, rng = carry
        rng, k_perm = jax.random.split(rng)

        # permutation & pre-gather to [M, B, ...]  (no indexing inside scan body)
        perm = jax.random.permutation(k_perm, num_full).reshape(M, B)  # [M, B]

        obs_mb = flat_obs[perm]  # [M, B, D]
        act_mb = flat_act[perm]  # [M, B]
        ret_mb = flat_ret[perm]  # [M, B]

        # scan over minibatches (leading axis M)
        (q_net, opt), losses = scan_minibatches(
            (q_net, opt),
            (obs_mb, act_mb, ret_mb),
        )
        # mean loss this epoch
        return (q_net, opt, rng), jnp.mean(losses)

    (q_net, opt, rng), epoch_losses = scan_epochs(
        (q_net, opt, rng), jnp.arange(params.update_epochs)
    )

    metrics = {"loss": jnp.mean(epoch_losses), "cum_return": jnp.mean(cum_ret)}
    return TrainState(q_net, opt, rng), metrics


def train_minibatched_jitted(
    q_net: Model, optimizer: nnx.Optimizer, config: EnvConfig, params: Params
):
    state = TrainState(q_net=q_net, opt=optimizer, rng=jax.random.PRNGKey(0))
    losses = []
    cum_returns = []
    for _ in range(params.num_iterations):
        state, metrics = _iteration_step(state, config, params)
        losses.append(metrics["loss"])
        cum_returns.append(metrics["cum_return"])
    return state.q_net, losses, cum_returns


def train_minibatched(
    q_net: Model,
    optimizer: nnx.Optimizer,
    config: EnvConfig,
    params: Params,
) -> tuple[Model, list, list]:
    rng_key = jax.random.PRNGKey(0)
    vmapped_rollout = jax.vmap(single_rollout, in_axes=(0, None, None, None, None))
    vmapped_q_lambda_return = jax.vmap(
        q_lambda_return, in_axes=(None, 0, 0, 0, None, None)
    )
    losses = []
    cum_returns = []
    times = {
        "rng_split": [],
        "rollout_ms": [],
        "q_lambda_return_ms": [],
        "reshape": [],
        "train_step": [],
        "update": [],
    }

    for iteration in tqdm(range(params.num_iterations)):
        split_pre = time.perf_counter()
        rng_keys = jax.random.split(rng_key, params.num_envs + 1)
        rng_key, rollout_keys = rng_keys[0], rng_keys[1:]
        times["rng_split"].append(time.perf_counter() - split_pre)

        rollout_pre = time.perf_counter()
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
        times["rollout_ms"].append(time.perf_counter() - rollout_pre)

        cum_returns.append(cum_return)

        q_lambda_return_pre = time.perf_counter()
        returns = vmapped_q_lambda_return(
            q_net,
            rewards_buffer,
            done_buffer,
            next_obs_buffer,
            params.gamma,
            params.q_lambda,
        )
        times["q_lambda_return_ms"].append(time.perf_counter() - q_lambda_return_pre)

        reshape_pre = time.perf_counter()
        flat_observations = obs_buffer.reshape(-1, obs_buffer.shape[-1])
        flat_actions = actions_buffer.reshape(-1)
        flat_returns = returns.reshape(-1)
        num_samples = flat_observations.shape[0]
        minibatch_size = num_samples // params.num_minibatches
        times["reshape"].append(time.perf_counter() - reshape_pre)

        update_pre = time.perf_counter()
        for epoch in range(params.update_epochs):
            rng_key, perm_key = jax.random.split(rng_key)
            permuted_indices = jax.random.permutation(perm_key, num_samples)

            for i in range(params.num_minibatches):
                start_idx = i * minibatch_size
                # Ensure that the last minibatch gets any remaining samples.
                end_idx = (
                    (i + 1) * minibatch_size
                    if i < params.num_minibatches - 1
                    else num_samples
                )
                minibatch_idx = permuted_indices[start_idx:end_idx]

                minibatch_obs = flat_observations[minibatch_idx]
                minibatch_actions = flat_actions[minibatch_idx]
                minibatch_returns = flat_returns[minibatch_idx]

                train_step_pre = time.perf_counter()
                _, _, loss = train_step(
                    q_net,
                    optimizer,
                    minibatch_obs,
                    minibatch_actions,
                    minibatch_returns,
                )
                times["train_step"].append(time.perf_counter() - train_step_pre)
                losses.append(loss)

        times["update"].append(time.perf_counter() - update_pre)

    return q_net, losses, cum_returns, times


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
    q_net = Model(width * height * 4, 256, width * height * 4, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(q_net, optax.adam(params.lr))
    train_minibatched(
        q_net,
        optimizer,
        config,
        params,
    )
