# math_engine.jl
# Core math engine for 3D Conway's Game of Life
# Contains the rules and state transition logic for a 3D cellular automaton.
#
# Standard 3D GoL rules (B3/S23 extended to 3D):
#   - Birth:  a dead cell with exactly 3 live neighbors becomes alive.
#   - Survival: a live cell with 2 or 3 live neighbors stays alive.
#   - Death:  a live cell with <2 or >3 live neighbors dies.
#
# In 3D, each cell has 26 neighbors (3x3x3 minus self).

module MathEngine

export Grid3D, create_grid, random_grid, count_neighbors, step!,
       BIRTH_RULE, SURVIVAL_RULE, DEFAULT_SIZE, DEFAULT_DENSITY

# --- Independent variables / rules ---

"""Birth rule: number of live neighbors required for a dead cell to become alive."""
const BIRTH_RULE = Set([3])

"""Survival rule: number of live neighbors required for a live cell to survive."""
const SURVIVAL_RULE = Set([2, 3])

"""Default grid size (cubic)."""
const DEFAULT_SIZE = 50

"""Default initial live-cell density (0.0 to 1.0)."""
const DEFAULT_DENSITY = 0.2

# --- Grid type ---

"""
A 3D grid of cells for the Game of Life.

Fields:
- `cells`: 3D BitArray where `true` = alive, `false` = dead.
- `size`:  the edge length of the cubic grid.
"""
mutable struct Grid3D
    cells::BitArray{3}
    size::Int
end

# --- Grid creation ---

"""Create an empty (all-dead) grid of the given size."""
function create_grid(size::Int = DEFAULT_SIZE)
    cells = falses(size, size, size)
    return Grid3D(cells, size)
end

"""Create a grid with random initial state at the given density."""
function random_grid(size::Int = DEFAULT_SIZE, density::Float64 = DEFAULT_DENSITY)
    cells = rand(size, size, size) .< density
    return Grid3D(cells, size)
end

# --- Neighbor counting ---

"""
Count the number of live neighbors for cell (x, y, z).

Uses periodic (toroidal) boundary conditions so the grid wraps around.
"""
function count_neighbors(grid::Grid3D, x::Int, y::Int, z::Int)
    n = 0
    s = grid.size
    for dx in -1:1, dy in -1:1, dz in -1:1
        (dx == 0 && dy == 0 && dz == 0) && continue
        nx = mod1(x + dx, s)
        ny = mod1(y + dy, s)
        nz = mod1(z + dz, s)
        n += Int(grid.cells[nx, ny, nz])
    end
    return n
end

# --- State transition ---

"""
Advance the grid by one generation in-place.

Applies the B3/S23 rules to every cell simultaneously.
"""
function step!(grid::Grid3D)
    s = grid.size
    # Compute neighbor counts for all cells
    neighbor_counts = zeros(Int, s, s, s)
    for x in 1:s, y in 1:s, z in 1:s
        neighbor_counts[x, y, z] = count_neighbors(grid, x, y, z)
    end

    # Apply rules
    for x in 1:s, y in 1:s, z in 1:s
        alive = grid.cells[x, y, z]
        n = neighbor_counts[x, y, z]
        if alive
            grid.cells[x, y, z] = n in SURVIVAL_RULE
        else
            grid.cells[x, y, z] = n in BIRTH_RULE
        end
    end
    return grid
end

"""
Return a 3D array of (x, y, z) coordinates for all live cells.
Useful for rendering — only alive cells need to be drawn.
"""
function live_cells(grid::Grid3D)
    s = grid.size
    coords = Tuple{Int,Int,Int}[]
    for x in 1:s, y in 1:s, z in 1:s
        if grid.cells[x, y, z]
            push!(coords, (x, y, z))
        end
    end
    return coords
end

"""Count total live cells in the grid."""
function population(grid::Grid3D)
    return count(grid.cells)
end

end # module
