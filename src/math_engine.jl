# math_engine.jl
# Core math engine for volumetric blob Life (continuous analog of 3D GoL).
#
# Space is a finite cube with no internal lattice. Entities are blobs; the
# tangible neighborhood is each blob's probability cloud:
#   cloud_radius = CLOUD_RADIUS_FACTOR × entity_diameter  (default 1.5×)
#
# Regular B3/S23 rules, applied to clouds:
#   - Survival: a blob with 2 or 3 other clouds overlapping its cloud stays.
#   - Birth:    a new blob appears at a triple-cloud intersection covered by
#               exactly 3 clouds (and not already occupied by a blob core).
#   - Death:    any other live blob is removed.
#
# Cloud–cloud overlap uses periodic (toroidal) distance so the volume stays
# finite without hard walls.
#
# Motion is continuous (integrate!): damped wandering plus steering toward
# the B3/S23 overlap band (about 2–3 neighboring clouds) and toward nearby
# triple-intersection birth sites.

module MathEngine

using LinearAlgebra

export BlobField, random_field, step!, integrate!, seed_reproducing_groups!,
       live_positions, population,
       BIRTH_RULE, SURVIVAL_RULE, DEFAULT_SIZE, DEFAULT_DENSITY,
       DEFAULT_DIAMETER, CLOUD_RADIUS_FACTOR,
       IDEAL_OVERLAP, TARGET_SEPARATION, MOVE_DAMPING, MOVE_NOISE,
       REPRO_ACCEL, SEPARATION_ACCEL, LONG_RANGE_ACCEL, MAX_SPEED, SEARCH_SIGMA,
       SEED_EVERY_TICKS, SEED_GROUPS

# --- Independent variables / rules ---

"""Birth rule: number of overlapping clouds required for a new blob."""
const BIRTH_RULE = Set([3])

"""Survival rule: number of overlapping clouds required to stay alive."""
const SURVIVAL_RULE = Set([2, 3])

"""Default edge length of the bounded volume."""
const DEFAULT_SIZE = 50.0

"""Default initial density (blobs per cubic world-unit)."""
const DEFAULT_DENSITY = 0.01

"""Default blob diameter in world units."""
const DEFAULT_DIAMETER = 1.0

"""Probability-cloud radius as a multiple of entity diameter."""
const CLOUD_RADIUS_FACTOR = 1.5

"""Preferred overlap count (midpoint of survival/birth)."""
const IDEAL_OVERLAP = 2.5

"""Preferred spacing inside a fertile cluster."""
const TARGET_SEPARATION = 1.7

const SEARCH_SIGMA = 3.0
const MOVE_DAMPING = 1.6
const MOVE_NOISE = 0.35
const REPRO_ACCEL = 1.4
const SEPARATION_ACCEL = 1.8
const LONG_RANGE_ACCEL = 0.45
const MAX_SPEED = 0.9
const SEED_EVERY_TICKS = 6
const SEED_GROUPS = 4

# --- Field type ---

"""
A formless but bounded volume of blobs.

Fields:
- `positions`:  N×3 array of blob centers.
- `velocities`: N×3 array of world-space velocities.
- `size`:       edge length of the cubic volume.
- `diameter`:   blob core diameter.
- `cloud_radius`: interaction radius (`CLOUD_RADIUS_FACTOR * diameter`).
"""
mutable struct BlobField
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    size::Float64
    diameter::Float64
    cloud_radius::Float64
end

cloud_radius(diameter::Float64) = CLOUD_RADIUS_FACTOR * diameter

# --- Creation ---

function random_field(size::Float64 = DEFAULT_SIZE,
                      density::Float64 = DEFAULT_DENSITY,
                      diameter::Float64 = DEFAULT_DIAMETER)
    n = max(0, round(Int, density * size^3))
    positions = rand(n, 3) .* size
    velocities = 0.12 .* randn(n, 3)
    return BlobField(positions, velocities, size, diameter, cloud_radius(diameter))
end

# --- Geometry ---

"""Shortest signed delta on a periodic interval of length `size`."""
function periodic_delta(d::Float64, size::Float64)
    return d - size * round(d / size)
end

function periodic_delta_vec(p, q, size::Float64)
    return (periodic_delta(p[1] - q[1], size),
            periodic_delta(p[2] - q[2], size),
            periodic_delta(p[3] - q[3], size))
end

function periodic_dist(p, q, size::Float64)
    dx, dy, dz = periodic_delta_vec(p, q, size)
    return sqrt(dx * dx + dy * dy + dz * dz)
end

"""Mean of three points using `origin` as the periodic unwrap anchor."""
function periodic_mean3(p, q, r, origin, size::Float64)
    acc = (0.0, 0.0, 0.0)
    for pt in (p, q, r)
        dx, dy, dz = periodic_delta_vec(pt, origin, size)
        acc = (acc[1] + origin[1] + dx,
               acc[2] + origin[2] + dy,
               acc[3] + origin[3] + dz)
    end
    return (mod(acc[1] / 3, size), mod(acc[2] / 3, size), mod(acc[3] / 3, size))
end

"""Circumcenter of three points in periodic space, or `nothing` if degenerate."""
function periodic_circumcenter(p, q, r, size::Float64)
    ax, ay, az = periodic_delta_vec(q, p, size)
    bx, by, bz = periodic_delta_vec(r, p, size)
    a = (ax, ay, az)
    b = (bx, by, bz)
    axb = (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    n2 = axb[1]^2 + axb[2]^2 + axb[3]^2
    n2 < 1e-12 && return nothing
    adot = ax * ax + ay * ay + az * az
    bdot = bx * bx + by * by + bz * bz
    t = (adot * b[1] - bdot * a[1],
         adot * b[2] - bdot * a[2],
         adot * b[3] - bdot * a[3])
    # rel = cross(t, axb) / (2 n2)
    rel = ((t[2] * axb[3] - t[3] * axb[2]) / (2 * n2),
           (t[3] * axb[1] - t[1] * axb[3]) / (2 * n2),
           (t[1] * axb[2] - t[2] * axb[1]) / (2 * n2))
    c = (mod(p[1] + rel[1], size), mod(p[2] + rel[2], size), mod(p[3] + rel[3], size))
    return c
end

function wrap_point(p, size::Float64)
    return (mod(p[1], size), mod(p[2], size), mod(p[3], size))
end

# --- Neighbor / coverage ---

function overlap_graph(field::BlobField)
    pos = field.positions
    n = size(pos, 1)
    R2 = 2 * field.cloud_radius
    neighbors = [Int[] for _ in 1:n]
    for i in 1:n
        pi = @view pos[i, :]
        for j in (i + 1):n
            pj = @view pos[j, :]
            if periodic_dist(pi, pj, field.size) < R2
                push!(neighbors[i], j)
                push!(neighbors[j], i)
            end
        end
    end
    return neighbors
end

function coverage_count(field::BlobField, point)
    pos = field.positions
    n = size(pos, 1)
    R = field.cloud_radius
    c = 0
    for i in 1:n
        if periodic_dist(@view(pos[i, :]), point, field.size) < R
            c += 1
        end
    end
    return c
end

function occupied(field::BlobField, point, extra_points)
    core_r = field.diameter / 2
    pos = field.positions
    n = size(pos, 1)
    for i in 1:n
        periodic_dist(@view(pos[i, :]), point, field.size) < core_r && return true
    end
    for q in extra_points
        periodic_dist(q, point, field.size) < field.diameter && return true
    end
    return false
end

# --- State transition ---

"""
Advance the field by one generation in-place using B3/S23 on probability clouds.
"""
function step!(field::BlobField)
    pos = field.positions
    n = size(pos, 1)
    if n == 0
        field.velocities = zeros(0, 3)
        return field
    end

    neighbors = overlap_graph(field)
    R = field.cloud_radius

    keep = falses(n)
    for i in 1:n
        keep[i] = length(neighbors[i]) in SURVIVAL_RULE
    end

    births = Tuple{Float64,Float64,Float64}[]

    for i in 1:n
        ni = neighbors[i]
        length(ni) < 2 && continue
        for a_idx in 1:length(ni)
            j = ni[a_idx]
            j > i || continue
            for b_idx in (a_idx + 1):length(ni)
                k = ni[b_idx]
                k > i || continue
                k in neighbors[j] || continue

                pi = @view pos[i, :]
                pj = @view pos[j, :]
                pk = @view pos[k, :]
                c = periodic_circumcenter(pi, pj, pk, field.size)
                spawn = c === nothing ? periodic_mean3(pi, pj, pk, pi, field.size) : c
                spawn = wrap_point(spawn, field.size)

                periodic_dist(spawn, pi, field.size) <= R + 1e-9 || continue
                periodic_dist(spawn, pj, field.size) <= R + 1e-9 || continue
                periodic_dist(spawn, pk, field.size) <= R + 1e-9 || continue
                coverage_count(field, spawn) == 3 || continue
                occupied(field, spawn, births) && continue
                push!(births, spawn)
            end
        end
    end

    kept = pos[keep, :]
    kept_vel = field.velocities[keep, :]
    if isempty(births)
        field.positions = kept
        field.velocities = kept_vel
    else
        born = Matrix{Float64}(undef, length(births), 3)
        for (idx, p) in enumerate(births)
            born[idx, 1] = p[1]
            born[idx, 2] = p[2]
            born[idx, 3] = p[3]
        end
        field.positions = vcat(kept, born)
        field.velocities = vcat(kept_vel, 0.12 .* randn(length(births), 3))
    end
    return field
end

"""
Advance positions with damped wandering and a bias toward reproducible overlap.
"""
function integrate!(field::BlobField, dt::Float64)
    pos = field.positions
    n = size(pos, 1)
    (n == 0 || dt <= 0) && return field

    R2 = 2 * field.cloud_radius
    accel = zeros(n, 3)
    counts = zeros(Int, n)
    deltas = Array{Float64}(undef, n, n, 3)
    dist = fill(Inf, n, n)

    for i in 1:n, j in 1:n
        i == j && continue
        dx, dy, dz = periodic_delta_vec(@view(pos[j, :]), @view(pos[i, :]), field.size)
        deltas[i, j, 1] = dx
        deltas[i, j, 2] = dy
        deltas[i, j, 3] = dz
        d = sqrt(dx * dx + dy * dy + dz * dz)
        dist[i, j] = d
        d < R2 && (counts[i] += 1)
    end

    sigma2 = 2 * SEARCH_SIGMA * SEARCH_SIGMA
    for i in 1:n
        need = IDEAL_OVERLAP - counts[i]
        isolate = counts[i] < 2
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in 1:n
            i == j && continue
            d = dist[i, j]
            inv = 1.0 / max(d, 1e-6)
            dx = deltas[i, j, 1]
            dy = deltas[i, j, 2]
            dz = deltas[i, j, 3]
            gap = (TARGET_SEPARATION - d) / TARGET_SEPARATION
            if gap > 0
                w = SEPARATION_ACCEL * gap * inv
                ax -= w * dx
                ay -= w * dy
                az -= w * dz
            end
            g = exp(-(d * d) / sigma2) * inv
            ax += REPRO_ACCEL * need * g * dx
            ay += REPRO_ACCEL * need * g * dy
            az += REPRO_ACCEL * need * g * dz
            if isolate
                lw = LONG_RANGE_ACCEL / (d * d + SEARCH_SIGMA * SEARCH_SIGMA) * inv
                ax += lw * dx
                ay += lw * dy
                az += lw * dz
            end
        end
        accel[i, 1] = ax
        accel[i, 2] = ay
        accel[i, 3] = az
    end

    for i in 1:n
        counts[i] != 2 && continue
        js = Int[]
        for j in 1:n
            (j != i && dist[i, j] < R2) && push!(js, j)
        end
        length(js) < 2 && continue
        pi = @view pos[i, :]
        pj = @view pos[js[1], :]
        pk = @view pos[js[2], :]
        c = periodic_circumcenter(pi, pj, pk, field.size)
        site = c === nothing ? periodic_mean3(pi, pj, pk, pi, field.size) : c
        dx, dy, dz = periodic_delta_vec(site, pi, field.size)
        accel[i, 1] += REPRO_ACCEL * dx
        accel[i, 2] += REPRO_ACCEL * dy
        accel[i, 3] += REPRO_ACCEL * dz
    end

    vel = field.velocities
    for i in 1:n
        vel[i, 1] += (accel[i, 1] - MOVE_DAMPING * vel[i, 1] + MOVE_NOISE * randn()) * dt
        vel[i, 2] += (accel[i, 2] - MOVE_DAMPING * vel[i, 2] + MOVE_NOISE * randn()) * dt
        vel[i, 3] += (accel[i, 3] - MOVE_DAMPING * vel[i, 3] + MOVE_NOISE * randn()) * dt
        sp = sqrt(vel[i, 1]^2 + vel[i, 2]^2 + vel[i, 3]^2)
        if sp > MAX_SPEED
            s = MAX_SPEED / sp
            vel[i, 1] *= s
            vel[i, 2] *= s
            vel[i, 3] *= s
        end
        pos[i, 1] = mod(pos[i, 1] + vel[i, 1] * dt, field.size)
        pos[i, 2] = mod(pos[i, 2] + vel[i, 2] * dt, field.size)
        pos[i, 3] = mod(pos[i, 3] + vel[i, 3] * dt, field.size)
    end
    return field
end

"""Inject compact 3-blob trios whose clouds already overlap (B3-ready)."""
function seed_reproducing_groups!(field::BlobField, groups::Int = SEED_GROUPS)
    side = TARGET_SEPARATION
    height = side * sqrt(3) / 2
    triangle = [
        0.0  (2 * height / 3)  0.0;
        (-0.5 * side)  (-height / 3)  0.0;
        (0.5 * side)  (-height / 3)  0.0
    ]
    new_pos = Matrix{Float64}(undef, groups * 3, 3)
    for g in 1:groups
        axis = randn(3)
        axis ./= (norm(axis) + 1e-12)
        hint = abs(axis[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
        a = cross(axis, hint)
        a ./= (norm(a) + 1e-12)
        b = cross(axis, a)
        rot = hcat(a, b, axis)
        center = rand(3) .* field.size
        for k in 1:3
            p = center .+ rot * triangle[k, :]
            new_pos[(g - 1) * 3 + k, 1] = mod(p[1], field.size)
            new_pos[(g - 1) * 3 + k, 2] = mod(p[2], field.size)
            new_pos[(g - 1) * 3 + k, 3] = mod(p[3], field.size)
        end
    end
    new_vel = 0.08 .* randn(groups * 3, 3)
    if size(field.positions, 1) == 0
        field.positions = new_pos
        field.velocities = new_vel
    else
        field.positions = vcat(field.positions, new_pos)
        field.velocities = vcat(field.velocities, new_vel)
    end
    return field
end

function live_positions(field::BlobField)
    return field.positions
end

function population(field::BlobField)
    return size(field.positions, 1)
end

end # module
