# simple-rts-rl


## Build docs

`rd -r build | uv run sphinx-build -b html ./source ./build | uv run python -m http.server -d build`

or

`uv run sphinx-autobuild source build/html`

**Simple-RTS-RL** is a minimalist grid-based real-time strategy game designed for reinforcement learning experiments. The game features two players competing to control territory by moving and battling on a rectangular grid.

---

## 🕹️ Game Overview

- **Grid Layout:** The game is played on a rectangular grid where each cell can hold troops.
- **Troop Types:**
  - **Player Troops:** Controlled by each player.
  - **Neutral Troops:** Populate specific cells and can be targeted.
  - **Bases:** Provide bonus troops over time.

---

## 📏 Game Rules

### 1. Movement

- Troops can move **Up**, **Right**, **Down**, or **Left** if the cell has more than 1 troop.
- Moves are restricted by the grid boundaries and the availability of troops.

### 2. Combat

- When troops move into a cell with enemy or neutral troops:
  - If attackers > defenders: The cell is captured.
  - If attackers ≤ defenders: The attack reduces the defender’s count.
- The source cell is always reduced to 1 troop.

### 3. Troop Reinforcement

- Troops are automatically reinforced over time.
- Cells marked as **bases** receive additional bonuses during reinforcement phases.

---
