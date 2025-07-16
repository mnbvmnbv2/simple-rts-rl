from typing import Sequence, Union
import math
import numpy as np
import matplotlib.pyplot as plt
import distinctipy
from IPython.display import clear_output

from src.rts.env import EnvState


def _draw_one_board(ax, state: EnvState) -> None:
    board = state.board
    h, w, n_players = board.height, board.width, board.num_players

    agent_colours = [list(c) for c in distinctipy.get_colors(n_players, rng=1)]

    def darken(col, k=0.6):  # adjust base‑square colour
        return [max(min(c * k, 1.0), 0.0) for c in col]

    agent_base_colours = [darken(c) for c in agent_colours]
    neutral_colour, neutral_base_colour, empty_colour = [0.5] * 3, [0.2] * 3, [1.0] * 3

    img = np.ones((h, w, 3))
    ax.set_xticks([])
    ax.set_yticks([])

    for y in range(h):
        for x in range(w):
            cell_drawn = False
            # Draw player troops
            for p in range(n_players):
                troops = int(board.player_troops[p, y, x])
                if troops > 0:
                    img[y, x] = (
                        agent_base_colours[p] if board.bases[y, x] else agent_colours[p]
                    )
                    ax.text(x, y, str(troops), ha="center", va="center", color="black")
                    cell_drawn = True
                    break
            # Draw neutral / empty
            if not cell_drawn:
                nt = int(board.neutral_troops[y, x])
                if nt > 0:
                    img[y, x] = (
                        neutral_base_colour if board.bases[y, x] else neutral_colour
                    )
                    ax.text(x, y, str(nt), ha="center", va="center", color="black")
                else:
                    img[y, x] = empty_colour
                    ax.text(x, y, "0", ha="center", va="center", color="black")

    ax.imshow(img)
    # Time stamp in the top‑left corner of each subplot
    ax.text(-0.5, -0.5, f"{state.time}", ha="center", va="center", color="purple")


def visualize_board(states: Union[EnvState, Sequence[EnvState]]) -> None:
    # Accept both a single state and an iterable of states transparently
    if isinstance(states, EnvState):
        states = [states]

    clear_output(wait=True)

    n = len(states)
    # Choose a reasonably square layout
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 4.5), squeeze=False
    )

    # Draw each board
    for idx, state in enumerate(states):
        r, c = divmod(idx, cols)
        _draw_one_board(axes[r, c], state)

    # Turn off any unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()
