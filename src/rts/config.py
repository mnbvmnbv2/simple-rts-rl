from flax import struct


@struct.dataclass
class EnvConfig:
    board_width: int
    board_height: int
    num_neutral_bases: int
    num_neutral_troops_start: int
    neutral_troops_min: int
    neutral_troops_max: int
    player_start_troops: int
    bonus_time: int
