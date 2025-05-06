import glob, re, time, signal, logging, jax, numpy as np, mlflow
from flax import nnx
import optax
from src.rl.pqn import Params, train_minibatched, Model
from src.rl.eval import evaluate_batch
from src.rts.config import EnvConfig, RewardConfig
from carbs import (
    CARBS,
    CARBSParams,
    ObservationInParam,
    Param,
    LogSpace,
    LogitSpace,
    LinearSpace,
)

MLFLOW_URI = "file:./mlruns"
EXPERIMENT = "carbs_indefinite"

param_spaces = [
    Param(
        name="num_iterations",
        space=LinearSpace(is_integer=True, min=1, max=5000, scale=100),
        search_center=40,
    ),
    Param(
        name="lr",
        space=LogSpace(scale=0.5),
        search_center=4e-4,
    ),
    Param(
        name="gamma",
        space=LogitSpace(),
        search_center=0.99,
    ),
    Param(
        name="q_lambda",
        space=LogitSpace(),
        search_center=0.92,
    ),
    Param(
        name="num_envs",
        space=LinearSpace(is_integer=True, min=1, max=5000, scale=100),
        search_center=200,
    ),
    Param(
        name="update_epochs",
        space=LinearSpace(is_integer=True, min=1, max=8, scale=4),
        search_center=1,
    ),
    Param(
        name="num_minibatches",
        space=LinearSpace(is_integer=True, min=1, max=16, scale=8),
        search_center=4,
    ),
    Param(
        name="epsilon",
        space=LogitSpace(),
        search_center=0.3,
    ),
]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s: %(message)s"
)
stop = False


def handler(sig, _):  # graceful Ctrl‑C / SIGTERM
    global stop
    stop = True


for s in (signal.SIGINT, signal.SIGTERM):
    signal.signal(s, handler)

ckpts = glob.glob("./checkpoints/carbs_experiment/carbs_*.pt")
if ckpts:
    # extract the trial number from filenames like "carbs_{N}obs.pt"
    def _num(f):
        m = re.search(r"carbs_(\d+)obs\.pt$", f)
        return int(m.group(1)) if m else -1

    latest = max(ckpts, key=_num)
    print(f"Loading CARBS state from {latest!r}")
    carbs = CARBS.load_from_file(latest)
    logging.info("Resumed CARBS with %d past observations", carbs.observation_count)
else:
    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )
    carbs = CARBS(carbs_params, param_spaces)
    logging.info("Started new CARBS run")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

width = height = 10
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
    reward_config=RewardConfig(),
)

trial_id = carbs.observation_count
while not stop:
    trial_id += 1
    sug = carbs.suggest().suggestion
    params = Params(
        num_iterations=sug["num_iterations"],
        lr=sug["lr"],
        gamma=sug["gamma"],
        q_lambda=sug["q_lambda"],
        num_envs=sug["num_envs"],
        num_steps=250,
        update_epochs=sug["update_epochs"],
        num_minibatches=sug["num_minibatches"],
        epsilon=sug["epsilon"],
    )

    t0 = time.time()
    qnet = Model(width * height * 4, 512, width * height * 4, rngs=nnx.Rngs(0))
    opt = nnx.Optimizer(qnet, optax.adam(params.lr))
    qnet, *_ = train_minibatched(qnet, opt, config, params)
    runtime = time.time() - t0
    output = float(
        np.mean(
            evaluate_batch(
                qnet, config, jax.random.PRNGKey(0), batch_size=100, num_steps=250
            )
        )
    )

    carbs.observe(ObservationInParam(input=sug, output=output, cost=runtime))

    with mlflow.start_run(run_name=f"trial_{trial_id}"):
        mlflow.log_params(sug)  # all H‑params
        mlflow.log_metric("output", output, step=trial_id)
        mlflow.log_metric("cost_sec", runtime, step=trial_id)

    logging.info("Trial %4d | output=%.4f cost=%.1fs", trial_id, output, runtime)
