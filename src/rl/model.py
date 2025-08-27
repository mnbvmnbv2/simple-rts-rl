from flax import nnx


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


if __name__ == "__main__":
    import time
    import jax
    import jax.numpy as jnp
    from flax import nnx

    # ---- params ----
    game_size = 10 * 10 * 4
    in_dim, mid_dim, out_dim = game_size, 512, game_size
    batch = 2048
    iters = 100
    key = jax.random.PRNGKey(0)

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

    # model and data
    model = Model(in_dim, mid_dim, out_dim, rngs=nnx.Rngs(key))
    x1 = jax.random.normal(key, (in_dim,))
    y1 = jax.random.normal(key, (out_dim,))
    X = jax.random.normal(key, (batch, in_dim))
    Y = jax.random.normal(key, (batch, out_dim))

    print(f"Device: {jax.devices()[0]}")
    print(f"Sizes — in:{in_dim} mid:{mid_dim} out:{out_dim} batch:{batch}")
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
