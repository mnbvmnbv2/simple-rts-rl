import distinctipy  # make sure to install via: pip install distinctipy
import jax
import jax.numpy as jnp
import pygame

from src.rts.config import EnvConfig
from src.rts.env import Board, EnvState, init_state, move, reinforce_troops
from src.rts.utils import do_random_move_for_player
from tests.helpers import assert_valid_state

# -------------------------------
# Configuration for GUI and Board
# -------------------------------
CELL_SIZE = 50
GRID_COLOR = (200, 200, 200)
BACKGROUND_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for selection
TEXT_COLOR = (0, 0, 0)

# Colors for neutrals and empty cells
NEUTRAL_COLOR = (128, 128, 128)
NEUTRAL_BASE_COLOR = (64, 64, 64)
EMPTY_COLOR = (255, 255, 255)


def draw_board(screen, board: Board, selected_cell=None):
    """Draw the board grid, troop counts, and highlight the selected cell.

    This version works for an arbitrary number of players (using board.player_troops
    with shape (num_players, height, width)) and generates distinct colors using distinctipy.
    """
    rows = board.height
    cols = board.width
    font = pygame.font.SysFont(None, 24)

    # Number of players from the board shape
    num_players = board.player_troops.shape[0]

    # Generate distinct colors for the agents.
    # distinctipy returns colors as floats in [0, 1]. Convert to 0-255 integer tuples.
    agent_colors_float = distinctipy.get_colors(num_players, rng=1)
    agent_colors = [tuple(int(c * 255) for c in color) for color in agent_colors_float]

    # Function to darken a color (for base cells)
    def darken_color(color, factor=0.6):
        return tuple(max(0, int(c * factor)) for c in color)

    agent_base_colors = [darken_color(color) for color in agent_colors]

    # Draw each cell.
    for row in range(rows):
        for col in range(cols):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = EMPTY_COLOR
            troop_count = 0

            # Determine which agent controls the cell (if any) by picking the one with the most troops.
            best_agent = None
            best_troops = 0
            for agent in range(num_players):
                count = board.player_troops[agent, row, col]
                if count > best_troops:
                    best_troops = count
                    best_agent = agent

            if best_troops > 0:
                # If the cell is a base, use the darker color variant.
                if board.bases[row, col]:
                    color = agent_base_colors[best_agent]
                else:
                    color = agent_colors[best_agent]
                troop_count = int(best_troops)
            elif board.neutral_troops[row, col] > 0:
                troop_count = int(board.neutral_troops[row, col])
                color = NEUTRAL_BASE_COLOR if board.bases[row, col] else NEUTRAL_COLOR
            else:
                color = EMPTY_COLOR

            # Draw the cell background and border.
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            # Render troop count text if nonzero.
            if troop_count:
                text_surface = font.render(str(troop_count), True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

            # Highlight the selected cell.
            if selected_cell == (row, col):
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)


def main():
    pygame.init()

    # Set up the game configuration and initial state.
    # Change num_players to any desired number.
    config = EnvConfig(
        num_players=4,  # e.g., 4 players
        board_width=10,
        board_height=10,
        num_neutral_bases=4,
        num_neutral_troops_start=8,
        neutral_troops_min=1,
        neutral_troops_max=10,
        player_start_troops=5,
        bonus_time=10,
    )
    rng_key = jax.random.PRNGKey(0)
    state: EnvState = init_state(rng_key, config)
    assert_valid_state(state)

    # Set up the display window.
    screen_width = config.board_width * CELL_SIZE
    screen_height = config.board_height * CELL_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Simple-RTS-RL Playable")
    clock = pygame.time.Clock()

    selected_cell = None  # Holds the (row, col) of the selected tile, if any.
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse clicks.
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0] // CELL_SIZE
                row = pos[1] // CELL_SIZE
                if 0 <= row < config.board_height and 0 <= col < config.board_width:
                    # Let user select the cell only if player 0 has more than 1 troop there.
                    if state.board.player_troops[0, row, col] > 1:
                        selected_cell = (row, col)
                    else:
                        selected_cell = None

            elif event.type == pygame.KEYDOWN:
                # Handle keyboard arrow keys for moves.
                if selected_cell is not None:
                    if event.key == pygame.K_UP:
                        action = 0
                    elif event.key == pygame.K_RIGHT:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_SPACE:
                        action = None
                    else:
                        continue

                    # Execute a move for player 0 (user-controlled) if an action is selected.
                    if action is not None:
                        sel_row, sel_col = selected_cell
                        state = move(
                            state, player=0, x=sel_col, y=sel_row, action=action
                        )

                    # For every opponent (players 1 to n-1), perform a random move.
                    (state, rng_key), _ = jax.lax.scan(
                        do_random_move_for_player,
                        (state, rng_key),
                        jnp.arange(1, config.num_players),
                    )

                    # Reinforce troops for all players.
                    state = reinforce_troops(state, config)
                    selected_cell = None

        # Render the updated board.
        screen.fill(BACKGROUND_COLOR)
        draw_board(screen, state.board, selected_cell)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
