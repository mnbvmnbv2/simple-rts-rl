from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
import distinctipy

from src.rts.env import EnvState


def visualize_board(state: EnvState) -> None:
    clear_output(wait=True)
    board = state.board
    num_agents = board.num_players

    agent_colors = distinctipy.get_colors(num_agents, rng=1)
    agent_colors = [list(color) for color in agent_colors]

    def darken_color(color, factor=0.6):
        return [max(min(c * factor, 1.0), 0.0) for c in color]

    agent_base_colors = [darken_color(color, factor=0.6) for color in agent_colors]

    neutral_color = [0.5, 0.5, 0.5]
    neutral_base_color = [0.2, 0.2, 0.2]
    empty_color = [1.0, 1.0, 1.0]

    image = np.ones((board.height, board.width, 3))

    for i in range(board.height):
        for j in range(board.width):
            cell_drawn = False

            for agent in range(num_agents):
                agent_troop_count = board.player_troops[agent, i, j]
                if agent_troop_count > 0:
                    if board.bases[i, j]:
                        image[i, j] = agent_base_colors[agent]
                    else:
                        image[i, j] = agent_colors[agent]
                    plt.text(
                        j,
                        i,
                        str(int(agent_troop_count)),
                        ha="center",
                        va="center",
                        color="black",
                    )
                    cell_drawn = True
                    break

            if not cell_drawn:
                if board.neutral_troops[i, j] > 0:
                    if board.bases[i, j]:
                        image[i, j] = neutral_base_color
                    else:
                        image[i, j] = neutral_color
                    plt.text(
                        j,
                        i,
                        str(int(board.neutral_troops[i, j])),
                        ha="center",
                        va="center",
                        color="black",
                    )
                else:
                    image[i, j] = empty_color
                    plt.text(j, i, "0", ha="center", va="center", color="black")

    # Display the current game time in the top left corner.
    plt.text(-0.5, -0.5, str(state.time), ha="center", va="center", color="purple")
    plt.axis("off")
    plt.imshow(image)
    plt.show()
