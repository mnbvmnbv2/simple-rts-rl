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
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="carbs_pqn")
def main(cfg: DictConfig):
    params = [
        (
            "num_iterations",
            LinearSpace(is_integer=True, min=1, max=5000, scale=100),
            40,
        ),
        ("lr", LogSpace(scale=0.5), 4e-4),
        ("gamma", LogitSpace(), 0.99),
        ("q_lambda", LogitSpace(), 0.92),
        # ("num_envs", LinearSpace(is_integer=True, min=1, max=5000, scale=100), 200),
        ("update_epochs", LinearSpace(is_integer=True, min=1, max=8, scale=4), 1),
        ("num_minibatches", LinearSpace(is_integer=True, min=1, max=16, scale=8), 4),
        ("epsilon", LogitSpace(), 0.3),
    ]
    param_spaces = [Param(name=p[0], space=p[1], search_center=p[2]) for p in params]

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

    mlflow.set_tracking_uri(cfg.MLFLOW_URI)
    mlflow.set_experiment(cfg.EXPERIMENT)

    reward_config = RewardConfig(
        tile_gain_reward=cfg.reward.tile_gain,
        tile_loss_reward=cfg.reward.tile_loss,
        base_gain_reward=cfg.reward.base_gain,
        base_loss_reward=cfg.reward.base_loss,
        victory_reward=cfg.reward.victory,
        defeat_reward=cfg.reward.defeat,
        neutral_tile_gain_reward=cfg.reward.neutral_tile_gain,
        opponent_tile_loss_reward=cfg.reward.opponent_tile_loss,
        opponent_tile_gain_reward=cfg.reward.opponent_tile_gain,
        opponent_base_loss_reward=cfg.reward.opponent_base_loss,
        opponent_base_gain_reward=cfg.reward.opponent_base_gain,
    )

    config = EnvConfig(
        num_players=cfg.env.num_players,
        board_width=cfg.env.width,
        board_height=cfg.env.height,
        num_neutral_bases=cfg.env.num_neutral_bases,
        num_neutral_troops_start=cfg.env.num_neutral_troops_start,
        neutral_troops_min=cfg.env.neutral_troops_min,
        neutral_troops_max=cfg.env.neutral_troops_max,
        player_start_troops=cfg.env.player_start_troops,
        bonus_time=cfg.env.bonus_time,
        reward_config=reward_config,
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
            num_envs=200,
            num_steps=250,
            update_epochs=sug["update_epochs"],
            num_minibatches=sug["num_minibatches"],
            epsilon=sug["epsilon"],
        )

        t0 = time.time()
        qnet = Model(
            cfg.env.width * cfg.env.height * 4,
            512,
            cfg.env.width * cfg.env.height * 4,
            rngs=nnx.Rngs(0),
        )
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


if __name__ == "__main__":
    main()
