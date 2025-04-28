from flax import struct


@struct.dataclass
class RewardConfig:
    tile_gain_reward: float = 1.0
    tile_loss_reward: float = -1.0
    base_gain_reward: float = 10.0
    base_loss_reward: float = -10.0
    victory_reward: float = 100.0
    defeat_reward: float = -100.0
    neutral_tile_gain_reward: float = 0.0
    opponent_tile_loss_reward: float = 0.0
    opponent_tile_gain_reward: float = -0.0
    opponent_base_loss_reward: float = 0.0
    opponent_base_gain_reward: float = -0.0


@struct.dataclass
class EnvConfig:
    num_players: int
    board_width: int
    board_height: int
    num_neutral_bases: int
    num_neutral_troops_start: int
    neutral_troops_min: int
    neutral_troops_max: int
    player_start_troops: int
    bonus_time: int
    reward_config: RewardConfig = RewardConfig()
