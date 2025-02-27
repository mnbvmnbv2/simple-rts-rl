from flax import struct


@struct.dataclass
class EnvConfig:
    board_width: int
    board_height: int
    num_neutral_bases: int
    num_neutral_troops_start: int
    neutral_bases_min_troops: int
    neutral_bases_max_troops: int
