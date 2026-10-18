"""
main_trial.py — 3D Conway's Game of Life rendering and variables.

This file handles:
  - Rendering a 50×50×50 grid of cells in 3D (software projection via pygame).
  - Rotatable view (mouse drag to rotate, scroll to zoom).
  - Tick system that advances the simulation.
  - Independent variables: grid size, density, rules, tick rate, colors.

The math engine (math_engine.jl) contains the core GoL rules.
This Python file mirrors those rules locally for rendering and can be
extended to call Julia via PyJulia when available.
"""

import sys
import math
import numpy as np
import pygame

# ──────────────────────────────────────────────
# Independent variables / configuration
# ──────────────────────────────────────────────

GRID_SIZE = 50          # Edge length of the cubic grid (cells per axis)
INITIAL_DENSITY = 0.01   # Probability a cell starts alive
TICK_RATE = 6          # Simulation ticks per second
FPS = 60                # Render frames per second

# Game of Life rules (B3/S23)
BIRTH_RULE = {3}        # Dead cell with exactly 3 neighbors → alive
SURVIVAL_RULE = {2, 3}  # Live cell with 2 or 3 neighbors → survives

# Rendering
WINDOW_SIZE = (800, 800)
BG_COLOR = (10, 10, 20)
GRID_COLOR = (40, 40, 50)
CELL_COLOR = (54, 54, 68)   # faint gray cell lattice
ALIVE_COLOR = (0, 220, 80)
DEAD_COLOR = (30, 30, 40)
BLOB_RADIUS = 4.0           # screen-space radius of a blob at scale 1
CAMERA_DISTANCE = 60.0
FOV = 60.0  # degrees

# ──────────────────────────────────────────────
# 3D Game of Life engine (mirrors math_engine.jl)
# ──────────────────────────────────────────────

class GoLEngine:
    """3D Conway's Game of Life engine using numpy for fast array ops."""

    def __init__(self, size=GRID_SIZE, density=INITIAL_DENSITY):
        self.size = size
        self.density = density
        self.grid = np.random.random((size, size, size)) < density
        self.generation = 0

    def count_neighbors(self):
        """Count live neighbors for every cell using toroidal boundary."""
        g = self.grid
        # Sum all 26 shifted versions of the grid
        neighbors = np.zeros_like(g, dtype=np.int32)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbors += np.roll(g, (dx, dy, dz), axis=(0, 1, 2))
        return neighbors

    def step(self):
        """Advance the simulation by one generation."""
        neighbors = self.count_neighbors()
        birth = np.isin(neighbors, list(BIRTH_RULE))
        survive = np.isin(neighbors, list(SURVIVAL_RULE))
        self.grid = (self.grid & survive) | (~self.grid & birth)
        self.generation += 1

    def live_cells(self):
        """Return list of (x, y, z) coordinates for all live cells."""
        coords = np.argwhere(self.grid)
        return [(int(x), int(y), int(z)) for x, y, z in coords]

    def population(self):
        """Return total number of live cells."""
        return int(np.sum(self.grid))


# ──────────────────────────────────────────────
# 3D rendering helpers
# ──────────────────────────────────────────────

def project_3d_to_2d(x, y, z, rot_x, rot_y, cam_dist, fov, win_w, win_h):
    """
    Project a 3D point onto 2D screen space.

    Applies rotation around X and Y axes, then perspective projection.
    """
    # Center the grid
    cx = x - GRID_SIZE / 2
    cy = y - GRID_SIZE / 2
    cz = z - GRID_SIZE / 2

    # Rotate around Y axis (horizontal drag)
    cos_y = math.cos(rot_y)
    sin_y = math.sin(rot_y)
    rx = cx * cos_y - cz * sin_y
    rz = cx * sin_y + cz * cos_y

    # Rotate around X axis (vertical drag)
    cos_x = math.cos(rot_x)
    sin_x = math.sin(rot_x)
    ry = cy * cos_x - rz * sin_x
    rz = cy * sin_x + rz * cos_x

    # Perspective projection
    f = fov / (fov + rz)
    screen_x = win_w / 2 + rx * f * cam_dist
    screen_y = win_h / 2 + ry * f * cam_dist
    return screen_x, screen_y, f


def draw_blob(surface, x, y, z, rot_x, rot_y, cam_dist, fov, win_w, win_h,
              alive=True):
    """Draw a small blob centered in the cell at grid position (x, y, z)."""
    sx, sy, scale = project_3d_to_2d(
        x + 0.5, y + 0.5, z + 0.5, rot_x, rot_y, cam_dist, fov, win_w, win_h
    )

    # Skip blobs behind the camera
    if scale < 0:
        return

    color = ALIVE_COLOR if alive else DEAD_COLOR
    size = max(1, int(scale * BLOB_RADIUS))
    pygame.draw.circle(surface, color, (int(sx), int(sy)), size)


def draw_cell_lattice(surface, rot_x, rot_y, cam_dist, fov, win_w, win_h):
    """Draw a faint gray lattice of cell boundaries on the far grid faces."""
    s = GRID_SIZE

    def far_plane(axis):
        """Return the coordinate of the face along `axis` farthest from camera."""
        near = [0, 0, 0]
        far = [0, 0, 0]
        far[axis] = s
        _, _, f_near = project_3d_to_2d(*near, rot_x, rot_y,
                                        cam_dist, fov, win_w, win_h)
        _, _, f_far = project_3d_to_2d(*far, rot_x, rot_y,
                                       cam_dist, fov, win_w, win_h)
        return 0 if f_near <= f_far else s

    px, py, pz = far_plane(0), far_plane(1), far_plane(2)

    def line(p1, p2):
        x1, y1, _ = project_3d_to_2d(*p1, rot_x, rot_y,
                                     cam_dist, fov, win_w, win_h)
        x2, y2, _ = project_3d_to_2d(*p2, rot_x, rot_y,
                                     cam_dist, fov, win_w, win_h)
        pygame.draw.line(surface, CELL_COLOR, (x1, y1), (x2, y2), 1)

    for i in range(s + 1):
        # Face x = px
        line((px, i, 0), (px, i, s))
        line((px, 0, i), (px, s, i))
        # Face y = py
        line((i, py, 0), (i, py, s))
        line((0, py, i), (s, py, i))
        # Face z = pz
        line((i, 0, pz), (i, s, pz))
        line((0, i, pz), (s, i, pz))


def draw_grid_wireframe(surface, rot_x, rot_y, cam_dist, fov, win_w, win_h):
    """Draw a faint wireframe of the grid boundaries for reference."""
    s = GRID_SIZE
    edges = []
    # 12 edges of the cube
    corners = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
               (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    edge_pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                  (0,4),(1,5),(2,6),(3,7)]
    for i, j in edge_pairs:
        edges.append((corners[i], corners[j]))

    for (x1, y1, z1), (x2, y2, z2) in edges:
        sx1, sy1, _ = project_3d_to_2d(x1, y1, z1, rot_x, rot_y,
                                       cam_dist, fov, win_w, win_h)
        sx2, sy2, _ = project_3d_to_2d(x2, y2, z2, rot_x, rot_y,
                                       cam_dist, fov, win_w, win_h)
        pygame.draw.line(surface, GRID_COLOR, (sx1, sy1), (sx2, sy2), 1)


# ──────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("3D Conway's Game of Life — Project CoDLEV")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Initialize engine
    engine = GoLEngine(size=GRID_SIZE, density=INITIAL_DENSITY)

    # Camera / rotation state
    rot_x = 0.3   # radians
    rot_y = 0.0
    cam_dist = CAMERA_DISTANCE
    fov = FOV

    # Interaction state
    dragging = False
    last_mouse = (0, 0)
    paused = False
    show_wireframe = True
    show_cells = True

    # Tick timing
    tick_accumulator = 0.0
    tick_interval = 1.0 / TICK_RATE

    print(f"Grid: {GRID_SIZE}³ | Density: {INITIAL_DENSITY} | "
          f"Rules: B{list(BIRTH_RULE)}/S{list(SURVIVAL_RULE)}")
    print(f"Initial population: {engine.population()}")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left drag
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:  # scroll up
                    cam_dist = max(5.0, cam_dist - 2.0)
                elif event.button == 5:  # scroll down
                    cam_dist += 2.0
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse[0]
                    dy = event.pos[1] - last_mouse[1]
                    rot_y += dx * 0.01
                    rot_x += dy * 0.01
                    rot_x = max(-math.pi / 2 + 0.1, min(math.pi / 2 - 0.1, rot_x))
                    last_mouse = event.pos
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    engine = GoLEngine(size=GRID_SIZE, density=INITIAL_DENSITY)
                elif event.key == pygame.K_w:
                    show_wireframe = not show_wireframe
                elif event.key == pygame.K_c:
                    show_cells = not show_cells
                elif event.key == pygame.K_EQUALS:
                    cam_dist = max(5.0, cam_dist - 5.0)
                elif event.key == pygame.K_MINUS:
                    cam_dist += 5.0

        # Simulation tick
        if not paused:
            tick_accumulator += dt
            if tick_accumulator >= tick_interval:
                engine.step()
                tick_accumulator -= tick_interval

        # Render
        screen.fill(BG_COLOR)

        if show_wireframe:
            draw_grid_wireframe(screen, rot_x, rot_y, cam_dist, fov,
                                *WINDOW_SIZE)

        if show_cells:
            draw_cell_lattice(screen, rot_x, rot_y, cam_dist, fov,
                              *WINDOW_SIZE)

        # Draw live cells
        for x, y, z in engine.live_cells():
            draw_blob(screen, x, y, z, rot_x, rot_y, cam_dist, fov,
                      *WINDOW_SIZE, alive=True)

        # Draw UI overlay
        status = "PAUSED" if paused else "RUNNING"
        info = [
            f"Gen: {engine.generation}  |  Pop: {engine.population()}",
            f"Status: {status}  |  Tick: {TICK_RATE} Hz",
            f"Rot: ({math.degrees(rot_x):.0f}°, {math.degrees(rot_y):.0f}°)",
            f"Cam dist: {cam_dist:.1f}",
            "Controls: [Space] pause  [R] random  [W] wireframe",
            "        [C] cells  [+/-] zoom  [drag] rotate",
        ]
        for i, line in enumerate(info):
            text = font.render(line, True, (200, 200, 220))
            screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
