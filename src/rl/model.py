from flax import nnx
import jax
import jax.numpy as jnp
from typing import Callable, Iterable


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Iterable[int],
        out_dim: int,
        *,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nnx.relu,
        use_layernorm: bool = True,
        pre_norm: bool = False,  # True: LN → Act; False: Act → LN
        dropout_rate: float = 0.0,  # 0 disables dropout
        residual: bool = False,  # add residuals when dims match
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.use_layernorm = use_layernorm
        self.pre_norm = pre_norm
        self.residual = residual
        self.dropout_rate = dropout_rate

        dims = [in_dim, *list(hidden_dims)]
        self.linears: list[nnx.Linear] = []
        self.norms: list[nnx.LayerNorm] = []
        self.drops: list[nnx.Dropout] = []

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            self.linears.append(nnx.Linear(d_in, d_out, rngs=rngs))
            if use_layernorm:
                self.norms.append(nnx.LayerNorm(d_out, rngs=rngs))
            else:
                self.norms.append(None)  # type: ignore
            if dropout_rate > 0.0:
                self.drops.append(nnx.Dropout(dropout_rate, rngs=rngs))
            else:
                self.drops.append(None)

        self.lin_out = nnx.Linear(dims[-1], out_dim, rngs=rngs)

    def __call__(self, x, *, training: bool = False):
        h = x
        for i, lin in enumerate(self.linears):
            y = lin(h)

            if self.use_layernorm and self.pre_norm:
                y = self.norms[i](y)  # type: ignore

            y = self.activation(y)

            if self.use_layernorm and not self.pre_norm:
                y = self.norms[i](y)  # type: ignore

            if self.drops[i] is not None:
                # deterministic=True disables dropout (i.e., eval mode)
                y = self.drops[i](y, deterministic=not training)  # type: ignore

            if self.residual and h.shape[-1] == y.shape[-1]:
                h = h + y
            else:
                h = y

        return self.lin_out(h)


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
