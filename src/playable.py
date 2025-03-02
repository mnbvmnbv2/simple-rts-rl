import pygame
import jax
import jax.numpy as jnp
import numpy as np

from src.rts.config import EnvConfig
from src.rts.env import init_state, move, reinforce_troops, EnvState, Board
from src.rts.utils import assert_valid_state, get_legal_moves, fixed_argwhere

# -------------------------------
# Configuration for GUI and Board
# -------------------------------
CELL_SIZE = 50
GRID_COLOR = (200, 200, 200)
BACKGROUND_COLOR = (255, 255, 255)
HIGHLIGHT_COLOR = (255, 255, 0)  # Yellow for selection
TEXT_COLOR = (0, 0, 0)

# Colors for players and neutrals (similar to visualization)
PLAYER_1_COLOR = (0, 0, 255)
PLAYER_1_BASE_COLOR = (0, 0, 180)
PLAYER_2_COLOR = (255, 0, 0)
PLAYER_2_BASE_COLOR = (180, 0, 0)
NEUTRAL_COLOR = (128, 128, 128)
NEUTRAL_BASE_COLOR = (64, 64, 64)
EMPTY_COLOR = (255, 255, 255)


def draw_board(screen, board: Board, selected_cell=None):
    """Draw the board grid, troop counts, and highlight the selected cell."""
    rows = board.height
    cols = board.width

    # Use a default font (pygame will cache this)
    font = pygame.font.SysFont(None, 24)

    for row in range(rows):
        for col in range(cols):
            # Compute the rectangle for the current cell.
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            # Determine cell color based on board state.
            color = EMPTY_COLOR
            if board.player_1_troops[row, col] > 0:
                color = PLAYER_1_BASE_COLOR if board.bases[row, col] else PLAYER_1_COLOR
            elif board.player_2_troops[row, col] > 0:
                color = PLAYER_2_BASE_COLOR if board.bases[row, col] else PLAYER_2_COLOR
            elif board.neutral_troops[row, col] > 0:
                color = NEUTRAL_BASE_COLOR if board.bases[row, col] else NEUTRAL_COLOR

            # Fill the cell
            pygame.draw.rect(screen, color, rect)
            # Draw a border for the cell
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

            # Determine troop count and draw it
            troop_count = 0
            if board.player_1_troops[row, col] > 0:
                troop_count = int(board.player_1_troops[row, col])
            elif board.player_2_troops[row, col] > 0:
                troop_count = int(board.player_2_troops[row, col])
            elif board.neutral_troops[row, col] > 0:
                troop_count = int(board.neutral_troops[row, col])

            if troop_count:
                text_surface = font.render(str(troop_count), True, TEXT_COLOR)
                text_rect = text_surface.get_rect(center=rect.center)
                screen.blit(text_surface, text_rect)

            # Highlight the selected cell
            if selected_cell == (row, col):
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)


def main():
    pygame.init()

    # Set up the game configuration and initial state
    config = EnvConfig(
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

    # Set up the display window
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

            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                col = pos[0] // CELL_SIZE
                row = pos[1] // CELL_SIZE
                if 0 <= row < config.board_height and 0 <= col < config.board_width:
                    # Try to select the clicked cell
                    if state.board.player_1_troops[row, col] > 1:
                        selected_cell = (row, col)
                    else:
                        selected_cell = None

            elif event.type == pygame.KEYDOWN:
                # Handle keyboard arrow keys for moves
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

                    if action is not None:
                        sel_row, sel_col = selected_cell
                        state = move(
                            state, player=0, x=sel_col, y=sel_row, action=action
                        )

                    # perform move for opponent
                    legal_actions_mask = get_legal_moves(state, 1)
                    legal_actions, num_actions = fixed_argwhere(
                        legal_actions_mask,
                        max_actions=state.board.width * state.board.height * 4,
                    )
                    rng_key, subkey = jax.random.split(rng_key)
                    action_idx = jax.random.randint(subkey, (), 0, num_actions)
                    action = jnp.take(legal_actions, action_idx, axis=0)
                    state = move(state, 1, action[1], action[0], action[2])

                    # Reinforce troops for both players
                    state = reinforce_troops(state, config)
                    selected_cell = None

        # Render the updated board
        screen.fill(BACKGROUND_COLOR)
        draw_board(screen, state.board, selected_cell)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
