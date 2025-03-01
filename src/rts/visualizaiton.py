from matplotlib import pyplot as plt
import numpy as np
from IPython.display import clear_output

from src.rts.env import EnvState


def visualize_board(state: EnvState) -> None:
    clear_output(wait=True)
    board = state.board
    # Visualize the board
    player_1_base_color = [0.0, 0.0, 0.8]
    player_1_color = [0.0, 0.0, 1.0]
    player_2_base_color = [0.8, 0.0, 0.0]
    player_2_color = [1.0, 0.0, 0.0]
    neutral_base_color = [0.2, 0.2, 0.2]
    neutral_color = [0.5, 0.5, 0.5]
    empty_color = [1.0, 1.0, 1.0]

    # Create a new image
    image = np.ones((board.width, board.height, 3))

    # Fill the image with the colors
    for i in range(board.height):
        for j in range(board.width):
            if board.player_1_troops[i, j] > 0:
                if board.bases[i, j]:
                    image[i, j] = player_1_base_color
                else:
                    image[i, j] = player_1_color
                plt.text(
                    j,
                    i,
                    str(int(board.player_1_troops[i, j])),
                    ha="center",
                    va="center",
                    color="black",
                )
            elif board.player_2_troops[i, j] > 0:
                if board.bases[i, j] > 0:
                    image[i, j] = player_2_base_color
                else:
                    image[i, j] = player_2_color
                plt.text(
                    j,
                    i,
                    str(int(board.player_2_troops[i, j])),
                    ha="center",
                    va="center",
                    color="black",
                )
            elif board.neutral_troops[i, j] > 0:
                if board.bases[i, j] > 0:
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

    # In top left corner, show the time as pink number
    plt.text(-0.5, -0.5, str(state.time), ha="center", va="center", color="purple")

    # remove the axis
    plt.axis("off")

    # Show the image
    plt.imshow(image)
    plt.show()
