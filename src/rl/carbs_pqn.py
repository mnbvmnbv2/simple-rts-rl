import glob
import re
import os
import logging
import jax
import numpy as np
import mlflow
from flax import nnx
import optax
from src.rl.model import Model
from src.rl.pqn import Params, train_minibatched
from src.rl.eval import evaluate_batch
from src.rts.config import EnvConfig, RewardConfig
from carbs import CARBS, CARBSParams, ObservationInParam, Param
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="carbs_pqn")
def main(cfg: DictConfig):
    params = []
    for p in cfg.params:
        space = instantiate(p.space)
        params.append((p.name, space, p.default))
    param_spaces = [Param(name=p[0], space=p[1], search_center=p[2]) for p in params]

    # This is used in CARBS
    os.environ["EXPERIMENT_NAME"] = cfg.EXPERIMENT

    ckpts = glob.glob(f"./checkpoints/{cfg.EXPERIMENT}/carbs_*.pt")
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
            # resample_frequency=0,
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
    for _ in range(cfg.num_trials):
        trial_id += 1
        sug = carbs.suggest().suggestion
        params = Params(
            num_iterations=sug["num_iterations"],
            lr=sug["lr"],
            gamma=sug["gamma"],
            q_lambda=sug["q_lambda"],
            num_envs=cfg.num_envs,
            num_steps=cfg.num_steps,
            update_epochs=sug["update_epochs"],
            num_minibatches=sug["num_minibatches"],
            epsilon=sug["epsilon"],
        )

        qnet = Model(
            cfg.env.width * cfg.env.height * 4,
            512,
            cfg.env.width * cfg.env.height * 4,
            rngs=nnx.Rngs(0),
        )
        opt = nnx.Optimizer(qnet, optax.adam(params.lr))
        qnet, _, _, times = train_minibatched(qnet, opt, config, params)
        # get time from after first iteration (JIT)
        runtime = sum(sum(v[1:]) for _, v in times.items() if len(v) > 1)
        output = float(
            np.mean(
                evaluate_batch(
                    qnet,
                    config,
                    jax.random.PRNGKey(0),
                    batch_size=cfg.num_envs,
                    num_steps=cfg.num_steps,
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
