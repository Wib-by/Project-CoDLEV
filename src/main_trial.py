"""
main_trial.py — volumetric blob Life rendering and variables.

This file handles:
  - Rendering blobs in a bounded, formless 3D volume (software projection).
  - Rotatable view (mouse drag to rotate, scroll to zoom).
  - Continuous motion with a bias toward B3/S23-fertile cloud overlap.
  - Tick system that advances reproduction.
  - Independent variables: volume size, density, diameter, rules, tick rate.

The math engine (math_engine.jl) contains the core cloud-overlap rules.
This Python file mirrors those rules locally for rendering and can be
extended to call Julia via PyJulia when available.
"""

import math
import sys

import numpy as np
import pygame

# ──────────────────────────────────────────────
# Independent variables / configuration
# ──────────────────────────────────────────────

SPACE_SIZE = 50.0           # Edge length of the bounded volume (world units)
INITIAL_DENSITY = 0.01      # Initial blobs per cubic world-unit
ENTITY_DIAMETER = 1.0       # Blob core diameter
CLOUD_RADIUS_FACTOR = 1.5   # Probability-cloud radius = factor × diameter
TICK_RATE = 0.5             # Simulation ticks per second (slow enough to watch)
FPS = 60                    # Render frames per second
SEED_EVERY_TICKS = 6        # Inject fertile groups this often
SEED_GROUPS = 4             # Trios per injection (3 blobs each, enough to reproduce)

# Game of Life rules (B3/S23), applied to probability-cloud overlap
BIRTH_RULE = {3}            # Empty region covered by exactly 3 clouds → birth
SURVIVAL_RULE = {2, 3}      # Blob whose cloud overlaps 2 or 3 others → survives

# Motion — wander plus a bias toward 2–3 cloud overlap (the B3/S23 band)
IDEAL_OVERLAP = 2.5         # Midpoint of survival/birth neighbor counts
TARGET_SEPARATION = 1.7     # Preferred spacing inside a fertile trio
SEARCH_SIGMA = 3.0          # Gaussian range for density-seeking (world units)
MOVE_DAMPING = 1.6          # Velocity decay (1/s)
MOVE_NOISE = 0.35           # Random acceleration scale
REPRO_ACCEL = 1.4           # Strength of density / birth-site steering
SEPARATION_ACCEL = 1.8      # Soft core spacing
LONG_RANGE_ACCEL = 0.45     # Isolates drift toward distant blobs
MAX_SPEED = 0.9             # World units per second

# Rendering
WINDOW_SIZE = (800, 800)
BG_COLOR = (10, 10, 20)
VOLUME_COLOR = (40, 40, 50)
ALIVE_COLOR = (0, 220, 80)
CLOUD_COLOR = (40, 160, 120)
CAMERA_DISTANCE = 12.0      # Start zoomed out so the full volume fits
FOV = 60.0  # degrees


# ──────────────────────────────────────────────
# Volumetric blob engine (mirrors math_engine.jl)
# ──────────────────────────────────────────────

class BlobEngine:
    """Bounded continuous volume of blobs with probability-cloud Life rules."""

    def __init__(self, size=SPACE_SIZE, density=INITIAL_DENSITY,
                 diameter=ENTITY_DIAMETER):
        self.size = float(size)
        self.diameter = float(diameter)
        self.cloud_radius = CLOUD_RADIUS_FACTOR * self.diameter
        n = max(0, int(round(density * self.size ** 3)))
        self.positions = np.random.random((n, 3)) * self.size
        self.velocities = np.random.normal(0.0, 0.12, (n, 3))
        self.generation = 0

    def _periodic_delta(self, delta):
        """Wrap differences onto the shortest periodic interval."""
        return delta - self.size * np.round(delta / self.size)

    def _periodic_dist(self, a, b):
        d = self._periodic_delta(a - b)
        return np.sqrt(np.sum(d * d, axis=-1))

    def _overlap_matrix(self):
        """Boolean N×N cloud–cloud overlap (diagonal False)."""
        pos = self.positions
        n = len(pos)
        if n == 0:
            return np.zeros((0, 0), dtype=bool)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt(np.sum(self._periodic_delta(delta) ** 2, axis=2))
        overlap = dist < (2.0 * self.cloud_radius)
        np.fill_diagonal(overlap, False)
        return overlap

    def _circumcenter(self, p, q, r):
        """Periodic circumcenter of three points, or None if degenerate."""
        a = self._periodic_delta(q - p)
        b = self._periodic_delta(r - p)
        axb = np.cross(a, b)
        n2 = np.dot(axb, axb)
        if n2 < 1e-12:
            return None
        rel = np.cross(np.dot(a, a) * b - np.dot(b, b) * a, axb) / (2.0 * n2)
        return np.mod(p + rel, self.size)

    def _periodic_mean(self, points, origin):
        acc = np.zeros(3)
        for pt in points:
            acc += origin + self._periodic_delta(pt - origin)
        return np.mod(acc / len(points), self.size)

    def _coverage_count(self, point):
        if len(self.positions) == 0:
            return 0
        return int(np.sum(self._periodic_dist(self.positions, point)
                          < self.cloud_radius))

    def _occupied(self, point, extra):
        core_r = self.diameter / 2.0
        if len(self.positions) and np.any(
            self._periodic_dist(self.positions, point) < core_r
        ):
            return True
        for q in extra:
            if self._periodic_dist(q, point) < self.diameter:
                return True
        return False

    def step(self):
        """Advance one generation using B3/S23 on probability clouds."""
        pos = self.positions
        n = len(pos)
        if n == 0:
            self.velocities = np.zeros((0, 3))
            self.generation += 1
            return

        overlap = self._overlap_matrix()
        counts = overlap.sum(axis=1)
        keep = np.isin(counts, list(SURVIVAL_RULE))

        births = []
        if 3 in BIRTH_RULE and n >= 3:
            R = self.cloud_radius
            neighbors = [np.flatnonzero(overlap[i]) for i in range(n)]
            for i in range(n):
                ni = neighbors[i]
                if len(ni) < 2:
                    continue
                js = ni[ni > i]
                for a_idx, j in enumerate(js):
                    linked = overlap[j]
                    for k in js[a_idx + 1:]:
                        if not linked[k]:
                            continue
                        p, q, r = pos[i], pos[j], pos[k]
                        spawn = self._circumcenter(p, q, r)
                        if spawn is None:
                            spawn = self._periodic_mean((p, q, r), p)
                        if (self._periodic_dist(spawn, p) > R + 1e-9
                                or self._periodic_dist(spawn, q) > R + 1e-9
                                or self._periodic_dist(spawn, r) > R + 1e-9):
                            continue
                        if self._coverage_count(spawn) != 3:
                            continue
                        if self._occupied(spawn, births):
                            continue
                        births.append(spawn)

        kept = pos[keep]
        kept_vel = self.velocities[keep]
        if births:
            born = np.asarray(births, dtype=float)
            self.positions = np.vstack([kept, born])
            birth_vel = np.random.normal(0.0, 0.12, (len(births), 3))
            self.velocities = np.vstack([kept_vel, birth_vel])
        else:
            self.positions = kept
            self.velocities = kept_vel
        self.generation += 1

    def integrate(self, dt):
        """Smooth motion: wander, keep spacing, steer toward reproducible overlap."""
        pos = self.positions
        n = len(pos)
        if n == 0:
            return
        dt = float(dt)
        if dt <= 0.0:
            return

        vec, dist = self._pairwise()
        overlap = dist < (2.0 * self.cloud_radius)
        np.fill_diagonal(overlap, False)
        counts = overlap.sum(axis=1).astype(np.float64)

        accel = (
            self._separation_force(vec, dist)
            + self._density_force(vec, dist, counts)
            + self._long_range_force(vec, dist, counts)
            + self._birth_site_force(pos, overlap, counts)
            + np.random.normal(0.0, MOVE_NOISE, (n, 3))
        )

        vel = self.velocities
        vel += (accel - MOVE_DAMPING * vel) * dt
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        too_fast = speeds > MAX_SPEED
        vel = np.where(too_fast, vel * (MAX_SPEED / np.maximum(speeds, 1e-9)), vel)
        self.velocities = vel
        self.positions = np.mod(pos + vel * dt, self.size)

    def _pairwise(self):
        """Vectors and distances from each blob i toward every blob j."""
        pos = self.positions
        vec = self._periodic_delta(pos[None, :, :] - pos[:, None, :])
        dist = np.sqrt(np.sum(vec * vec, axis=2))
        np.fill_diagonal(dist, np.inf)
        return vec, dist

    def _unit_weights(self, vec, dist, weight):
        inv = 1.0 / np.maximum(dist, 1e-6)
        w = np.where(np.isfinite(dist), weight * inv, 0.0)
        return (vec * w[..., None]).sum(axis=1)

    def _separation_force(self, vec, dist):
        gap = (TARGET_SEPARATION - dist) / TARGET_SEPARATION
        weight = np.clip(gap, 0.0, 1.0)
        return -SEPARATION_ACCEL * self._unit_weights(vec, dist, weight)

    def _density_force(self, vec, dist, counts):
        """Seek the 2–3 overlap band: lonely blobs approach, crowded ones ease off."""
        need = (IDEAL_OVERLAP - counts)[:, None, None]
        sigma2 = 2.0 * SEARCH_SIGMA * SEARCH_SIGMA
        gauss = np.exp(-(dist * dist) / sigma2)
        gauss = np.where(np.isfinite(dist), gauss, 0.0)
        return REPRO_ACCEL * (vec * (gauss / np.maximum(dist, 1e-6))[..., None]
                              * need).sum(axis=1)

    def _long_range_force(self, vec, dist, counts):
        """Isolates slowly drift toward anyone they can still 'smell'."""
        isolate = (counts < 2.0).astype(np.float64)[:, None]
        if not np.any(isolate):
            return np.zeros_like(self.positions)
        falloff = 1.0 / (dist * dist + SEARCH_SIGMA * SEARCH_SIGMA)
        falloff = np.where(np.isfinite(dist), falloff, 0.0)
        pull = self._unit_weights(vec, dist, falloff)
        return LONG_RANGE_ACCEL * pull * isolate

    def _birth_site_force(self, pos, overlap, counts):
        """Nudge nearly-fertile pairs toward their triple intersection."""
        force = np.zeros_like(pos)
        almost = np.flatnonzero(counts == 2)
        if len(almost) == 0:
            return force
        for i in almost:
            js = np.flatnonzero(overlap[i])
            if len(js) < 2:
                continue
            site = self._circumcenter(pos[i], pos[js[0]], pos[js[1]])
            if site is None:
                site = self._periodic_mean((pos[i], pos[js[0]], pos[js[1]]), pos[i])
            force[i] += REPRO_ACCEL * self._periodic_delta(site - pos[i])
        return force

    def seed_reproducing_groups(self, groups=SEED_GROUPS):
        """Drop compact 3-blob trios whose clouds already overlap (B3-ready)."""
        side = TARGET_SEPARATION
        height = side * math.sqrt(3.0) / 2.0
        local = np.array([
            [0.0, 2.0 * height / 3.0, 0.0],
            [-0.5 * side, -height / 3.0, 0.0],
            [0.5 * side, -height / 3.0, 0.0],
        ])
        new_pos = np.empty((groups * 3, 3))
        for g in range(groups):
            axis = np.random.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            hint = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            a = np.cross(axis, hint)
            a /= np.linalg.norm(a) + 1e-12
            b = np.cross(axis, a)
            rot = np.column_stack((a, b, axis))
            center = np.random.random(3) * self.size
            new_pos[g * 3:(g + 1) * 3] = np.mod(center + local @ rot.T, self.size)
        new_vel = np.random.normal(0.0, 0.08, new_pos.shape)
        if len(self.positions) == 0:
            self.positions = new_pos
            self.velocities = new_vel
        else:
            self.positions = np.vstack([self.positions, new_pos])
            self.velocities = np.vstack([self.velocities, new_vel])

    def live_positions(self):
        return self.positions

    def population(self):
        return int(len(self.positions))


# ──────────────────────────────────────────────
# 3D rendering helpers
# ──────────────────────────────────────────────

def make_cloud_sprite(pixel_size=128, color=CLOUD_COLOR, max_alpha=72):
    """Radial falloff sprite used for probability clouds."""
    surf = pygame.Surface((pixel_size, pixel_size), pygame.SRCALPHA)
    cx = cy = pixel_size / 2.0
    rmax = pixel_size / 2.0 - 1.0
    y, x = np.ogrid[:pixel_size, :pixel_size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / rmax
    falloff = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    alpha = (falloff * max_alpha).astype(np.uint8)
    alpha[dist > 1.0] = 0
    rgb = np.zeros((pixel_size, pixel_size, 3), dtype=np.uint8)
    rgb[..., 0] = color[0]
    rgb[..., 1] = color[1]
    rgb[..., 2] = color[2]
    rgb_px = pygame.surfarray.pixels3d(surf)
    rgb_px[:, :, :] = np.transpose(rgb, (1, 0, 2))
    del rgb_px
    alpha_px = pygame.surfarray.pixels_alpha(surf)
    alpha_px[:, :] = alpha.T
    del alpha_px
    return surf


def project_3d_to_2d(x, y, z, rot_x, rot_y, cam_dist, fov, win_w, win_h):
    """
    Project a 3D point onto 2D screen space.

    Applies rotation around X and Y axes, then perspective projection.
    """
    cx = x - SPACE_SIZE / 2
    cy = y - SPACE_SIZE / 2
    cz = z - SPACE_SIZE / 2

    cos_y = math.cos(rot_y)
    sin_y = math.sin(rot_y)
    rx = cx * cos_y - cz * sin_y
    rz = cx * sin_y + cz * cos_y

    cos_x = math.cos(rot_x)
    sin_x = math.sin(rot_x)
    ry = cy * cos_x - rz * sin_x
    rz = cy * sin_x + rz * cos_x

    f = fov / (fov + rz)
    screen_x = win_w / 2 + rx * f * cam_dist
    screen_y = win_h / 2 + ry * f * cam_dist
    return screen_x, screen_y, f, rz


def draw_volume_wireframe(surface, rot_x, rot_y, cam_dist, fov, win_w, win_h):
    """Draw the bounding volume only — no internal lattice."""
    s = SPACE_SIZE
    corners = [(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0),
               (0, 0, s), (s, 0, s), (s, s, s), (0, s, s)]
    edge_pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edge_pairs:
        x1, y1, z1 = corners[i]
        x2, y2, z2 = corners[j]
        sx1, sy1, _, _ = project_3d_to_2d(x1, y1, z1, rot_x, rot_y,
                                          cam_dist, fov, win_w, win_h)
        sx2, sy2, _, _ = project_3d_to_2d(x2, y2, z2, rot_x, rot_y,
                                          cam_dist, fov, win_w, win_h)
        pygame.draw.line(surface, VOLUME_COLOR, (sx1, sy1), (sx2, sy2), 1)


_cloud_scale_cache = {}


def _scaled_cloud(sprite, radius_px):
    """Reuse scaled cloud sprites; radius is quantized to keep the cache small."""
    radius_px = max(2, min(int(radius_px), 400))
    radius_px = (radius_px + 1) & ~1
    cached = _cloud_scale_cache.get(radius_px)
    if cached is None:
        cached = pygame.transform.smoothscale(sprite, (radius_px * 2, radius_px * 2))
        if len(_cloud_scale_cache) > 96:
            _cloud_scale_cache.pop(next(iter(_cloud_scale_cache)))
        _cloud_scale_cache[radius_px] = cached
    return cached, radius_px


def draw_blobs(surface, positions, cloud_sprite, rot_x, rot_y, cam_dist, fov,
               win_w, win_h, show_clouds, diameter, cloud_radius):
    """Depth-sort and draw probability clouds, then blob cores."""
    projected = []
    for x, y, z in positions:
        sx, sy, scale, depth = project_3d_to_2d(
            x, y, z, rot_x, rot_y, cam_dist, fov, win_w, win_h
        )
        if scale < 0:
            continue
        projected.append((depth, sx, sy, scale))
    projected.sort(reverse=True)

    core_world_r = diameter / 2.0
    for _, sx, sy, scale in projected:
        px = int(sx)
        py = int(sy)
        if show_clouds:
            cloud_px = scale * cam_dist * cloud_radius
            sprite, cloud_px = _scaled_cloud(cloud_sprite, cloud_px)
            surface.blit(sprite, (px - cloud_px, py - cloud_px))
        core_px = max(1, int(scale * cam_dist * core_world_r))
        pygame.draw.circle(surface, ALIVE_COLOR, (px, py), core_px)


# ──────────────────────────────────────────────
# Main application
# ──────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Volumetric Blob Life — Project CoDLEV")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    cloud_sprite = make_cloud_sprite()

    engine = BlobEngine(size=SPACE_SIZE, density=INITIAL_DENSITY,
                        diameter=ENTITY_DIAMETER)

    rot_x = 0.3
    rot_y = 0.0
    cam_dist = CAMERA_DISTANCE
    fov = FOV

    dragging = False
    last_mouse = (0, 0)
    paused = False
    show_wireframe = True
    show_clouds = True

    tick_accumulator = 0.0
    tick_interval = 1.0 / TICK_RATE

    print(f"Volume: {SPACE_SIZE:.0f}³  |  Density: {INITIAL_DENSITY}  |  "
          f"Diameter: {ENTITY_DIAMETER}  |  Cloud r: {engine.cloud_radius}")
    print(f"Rules: B{sorted(BIRTH_RULE)}/S{sorted(SURVIVAL_RULE)} on cloud overlap")
    print(f"Tick: {TICK_RATE} Hz  |  Seed: {SEED_GROUPS} trios every {SEED_EVERY_TICKS} ticks")
    print(f"Initial population: {engine.population()}")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    cam_dist = max(5.0, cam_dist - 2.0)
                elif event.button == 5:
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
                    engine = BlobEngine(size=SPACE_SIZE, density=INITIAL_DENSITY,
                                        diameter=ENTITY_DIAMETER)
                elif event.key == pygame.K_w:
                    show_wireframe = not show_wireframe
                elif event.key == pygame.K_c:
                    show_clouds = not show_clouds
                elif event.key == pygame.K_EQUALS:
                    cam_dist = max(5.0, cam_dist - 5.0)
                elif event.key == pygame.K_MINUS:
                    cam_dist += 5.0

        if not paused:
            engine.integrate(dt)
            tick_accumulator += dt
            if tick_accumulator >= tick_interval:
                engine.step()
                if engine.generation % SEED_EVERY_TICKS == 0:
                    engine.seed_reproducing_groups()
                tick_accumulator -= tick_interval

        screen.fill(BG_COLOR)

        if show_wireframe:
            draw_volume_wireframe(screen, rot_x, rot_y, cam_dist, fov,
                                  *WINDOW_SIZE)

        draw_blobs(screen, engine.live_positions(), cloud_sprite,
                   rot_x, rot_y, cam_dist, fov, *WINDOW_SIZE,
                   show_clouds, engine.diameter, engine.cloud_radius)

        status = "PAUSED" if paused else "RUNNING"
        info = [
            f"Gen: {engine.generation}  |  Pop: {engine.population()}",
            f"Status: {status}  |  Tick: {TICK_RATE} Hz  |  Seed: every {SEED_EVERY_TICKS}",
            f"Cloud r: {engine.cloud_radius:.2f}  ({CLOUD_RADIUS_FACTOR}× diameter)",
            f"Rot: ({math.degrees(rot_x):.0f}°, {math.degrees(rot_y):.0f}°)"
            f"  Cam: {cam_dist:.1f}",
            "Controls: [Space] pause  [R] random  [W] bounds",
            "          [C] clouds  [+/-] zoom  [drag] rotate",
        ]
        for i, line in enumerate(info):
            text = font.render(line, True, (200, 200, 220))
            screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
