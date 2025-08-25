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
