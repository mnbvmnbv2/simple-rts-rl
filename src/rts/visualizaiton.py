from matplotlib import pyplot as plt
import numpy as np

from src.rts.env import EnvState


def visualize_board(state: EnvState) -> None:
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
    image = np.ones((board.shape[0], board.shape[1], 3))

    # Fill the image with the colors
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j, 0] > 0:
                if board[i, j, 3] > 0:
                    image[i, j] = player_1_base_color
                else:
                    image[i, j] = player_1_color
                plt.text(
                    j,
                    i,
                    str(int(board[i, j, 0])),
                    ha="center",
                    va="center",
                    color="black",
                )
            elif board[i, j, 1] > 0:
                if board[i, j, 4] > 0:
                    image[i, j] = player_2_base_color
                else:
                    image[i, j] = player_2_color
                plt.text(
                    j,
                    i,
                    str(int(board[i, j, 1])),
                    ha="center",
                    va="center",
                    color="black",
                )
            elif board[i, j, 2] > 0:
                if board[i, j, 5] > 0:
                    image[i, j] = neutral_base_color
                else:
                    image[i, j] = neutral_color
                plt.text(
                    j,
                    i,
                    str(int(board[i, j, 2])),
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
