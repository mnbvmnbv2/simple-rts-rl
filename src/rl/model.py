from typing import Callable, Iterable

import jax
import jax.numpy as jnp
from flax import nnx

DType = jnp.dtype


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        *,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        use_layernorm: bool = True,
        pre_norm: bool = False,
        dropout_rate: float = 0.0,
        residual: bool = False,
        param_dtype: DType = jnp.float16,
        compute_dtype: DType = jnp.float16,
        ln_compute_dtype: DType = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.pre_norm = pre_norm
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.param_dtype = param_dtype
        self.compute_dtype = compute_dtype
        self.ln_compute_dtype = ln_compute_dtype

        dims = [in_dim, *list(hidden_dims)]
        self.linears: list[nnx.Linear] = []
        self.norms: list[nnx.LayerNorm] = []
        self.drops: list[nnx.Dropout] = []

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            self.linears.append(nnx.Linear(d_in, d_out, rngs=rngs, dtype=param_dtype))
            if use_layernorm:
                self.norms.append(nnx.LayerNorm(d_out, rngs=rngs, dtype=jnp.float32))
            else:
                self.norms.append(None)  # type: ignore

            self.drops.append(
                nnx.Dropout(dropout_rate, rngs=rngs) if dropout_rate > 0 else None
            )

        self.lin_out = nnx.Linear(dims[-1], out_dim, rngs=rngs, dtype=param_dtype)

    def __call__(self, x, *, training: bool = False):
        h = x.astype(self.compute_dtype)

        for i, lin in enumerate(self.linears):
            y = lin(h).astype(self.compute_dtype)

            if self.use_layernorm and self.pre_norm:
                y = self.norms[i](y.astype(self.ln_compute_dtype)).astype(
                    self.compute_dtype
                )

            y = self.activation(y)

            if self.use_layernorm and not self.pre_norm:
                y = self.norms[i](y.astype(self.ln_compute_dtype)).astype(
                    self.compute_dtype
                )  # type: ignore

            if self.drops[i] is not None:
                y = self.drops[i](y, deterministic=not training)  # type: ignore

            if self.residual and h.shape[-1] == y.shape[-1]:
                h = (h + y).astype(self.compute_dtype)
            else:
                h = y

        out = self.lin_out(h).astype(self.compute_dtype)
        return out


class CNN(nnx.Module):
    def __init__(self, in_channels: int, action_space: int, rngs: nnx.Rngs):
        # 1. Convolutional feature extractor
        self.conv1 = nnx.Conv(
            in_channels, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(64, 64, kernel_size=(3, 3), padding="SAME", rngs=rngs)

        # 2. Flatten
        self.linear1 = nnx.Linear(64 * 10 * 10, 512, rngs=rngs)  # Assuming 10x10 board
        self.linear_out = nnx.Linear(512, action_space, rngs=rngs)

        self.relu = nnx.relu

    def __call__(self, x):
        # x shape: (height, width, channels)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten
        if x.ndim == 3:
            # Case 1: Unbatched (inside vmap) -> (H, W, C)
            x = x.reshape((-1,))  # Flatten everything -> (H*W*C,)
        else:
            # Case 2: Batched (training) -> (B, H, W, C)
            x = x.reshape((x.shape[0], -1))
        x = self.relu(self.linear1(x))
        x = self.linear_out(x)
        return x


if __name__ == "__main__":
    import time

    import jax
    import jax.numpy as jnp
    from flax import nnx

    # jax.default_matmul_precision("tensorfloat32")

    # ---- params ----
    key = jax.random.PRNGKey(0)
    game_size = 10 * 10 * 4
    in_dim, out_dim = game_size, game_size
    model = MLP(
        in_dim,
        [512, 512],
        out_dim,
        rngs=nnx.Rngs(key),
    )

    batch = 2048
    iters = 100

    # comment in/out:
    # nnx.display(model)

    def _sync(x):
        # Make sure device work is finished before timing
        jax.tree_util.tree_map(jax.block_until_ready, x)

    def bench(fn, *args, warmup=10, iters=50):
        for _ in range(warmup):
            _sync(fn(*args))
        t0 = time.perf_counter()
        for _ in range(iters):
            _sync(fn(*args))
        t1 = time.perf_counter()
        return iters / (t1 - t0)

    def loss_single(m: nnx.Module, x, y):
        pred = m(x)
        return jnp.mean((pred - y) ** 2)

    def loss_batched(m: nnx.Module, X, Y):
        preds = jax.vmap(lambda x: m(x))(X)
        return jnp.mean((preds - Y) ** 2)

    # Forwards
    fwd_single = nnx.jit(lambda m, x: m(x))
    fwd_batched = nnx.jit(lambda m, X: jax.vmap(lambda x: m(x))(X))

    # Backwards
    grad_single = nnx.jit(nnx.grad(loss_single))
    grad_batched = nnx.jit(nnx.grad(loss_batched))

    # data
    x1 = jax.random.normal(key, (in_dim,))
    y1 = jax.random.normal(key, (out_dim,))
    X = jax.random.normal(key, (batch, in_dim))
    Y = jax.random.normal(key, (batch, out_dim))

    print(f"Device: {jax.devices()[0]}")
    print("Warming up (compile excluded from timing)…")

    # forward
    sps = bench(fwd_single, model, x1, iters=iters)
    print(f"Forward single: {sps:.2f} steps/s | {sps:.2f} examples/s")

    sps_b = bench(fwd_batched, model, X, iters=iters)
    print(
        f"Forward batched (vmap): {sps_b:.2f} steps/s | {sps_b * batch:.0f} examples/s"
    )

    # backward
    # fewer iters since backward is heavier
    grad_iters = max(10, iters // 2)

    bps = bench(grad_single, model, x1, y1, iters=grad_iters)
    print(f"Backward single: {bps:.2f} steps/s | {bps:.2f} examples/s")

    bps_b = bench(grad_batched, model, X, Y, iters=grad_iters)
    print(
        f"Backward batched (vmap): {bps_b:.2f} steps/s | {bps_b * batch:.0f} examples/s"
    )

    def sweep_batches(
        model, in_dim, out_dim, batch_candidates, iters=30, key=jax.random.PRNGKey(0)
    ):
        # jitted once; retraces when batch shape changes (fine for a quick sweep)
        fwd_batched = nnx.jit(lambda m, X: jax.vmap(lambda x: m(x))(X))

        def loss_batched(m, X, Y):
            P = jax.vmap(lambda x: m(x))(X)
            return jnp.mean((P - Y) ** 2)

        bwd_batched = nnx.jit(nnx.grad(loss_batched))

        print("Batch sweep (forward/backward):")
        best = {"batch": None, "ex_s": -1, "f_sps": 0, "b_sps": 0}

        for b in batch_candidates:
            try:
                kx, ky = jax.random.split(key)
                X = jax.random.normal(kx, (b, in_dim))
                Y = jax.random.normal(ky, (b, out_dim))

                # compile + warmup
                _sync(fwd_batched(model, X))
                _sync(bwd_batched(model, X, Y))

                f_sps = bench(fwd_batched, model, X, iters=iters)
                b_sps = bench(bwd_batched, model, X, Y, iters=max(10, iters // 2))
                ex_s = (
                    b * f_sps
                )  # forward ex/s; you can also report backward ex/s as b*b_sps

                print(
                    f"  b={b:6d} | fwd {f_sps:8.0f} steps/s ({ex_s:10.0f} ex/s) | "
                    f"bwd {b_sps:8.0f} steps/s ({b * b_sps:10.0f} ex/s)"
                )

                if ex_s > best["ex_s"]:
                    best = {"batch": b, "ex_s": ex_s, "f_sps": f_sps, "b_sps": b_sps}

            except Exception as e:
                msg = str(e)
                if (
                    "RESOURCE_EXHAUSTED" in msg
                    or "OutOfMemory" in msg
                    or "out of memory" in msg.lower()
                ):
                    print(f"  b={b:6d} | OOM")
                    continue
                else:
                    print(f"  b={b:6d} | error: {e}")
                    continue

        print(
            f"\nBest by forward ex/s: batch={best['batch']} "
            f"→ {best['ex_s']:,.0f} ex/s (fwd {best['f_sps']:,.0f} steps/s, "
            f"bwd {best['b_sps']:,.0f} steps/s)"
        )
        return best

    batches = [2**k for k in range(6, 16)]  # 64..32768
    _ = sweep_batches(model, in_dim, out_dim, batches, iters=30)
