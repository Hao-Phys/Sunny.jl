struct RealSpaceQuarticVerticesSUN
    V41 :: Array{ComplexF64, 4}
    V42 :: Array{ComplexF64, 2}
    V43 :: Array{ComplexF64, 2}
end

struct RealSpaceQuarticVerticesDipole
    V41 :: ComplexF64
    V42 :: ComplexF64
    V43 :: ComplexF64
end

struct RealSpaceCubicVerticesSUN
    V31_p :: Vector{ComplexF64}
    V31_m :: Vector{ComplexF64}
    V32_p :: Array{ComplexF64, 3}
    V32_m :: Array{ComplexF64, 3}
end

struct RealSpaceCubicVerticesDipole
    V31 :: ComplexF64
    V32 :: ComplexF64
end

# The :tensormode will probably be disregarded in the future. Now if we use @tensor, there might be overhead of allocating new memory. In that case we can use the :loop mode. Otherwise, we can use the :tensor mode.
struct NonPerturbativeTheory
    swt :: SpinWaveTheory
    clustersize :: NTuple{3, Int}   # Cluster size for number of magnetic unit cell
    qs :: Array{Vec3, 3}
    Es :: Array{Float64, 4}
    Vps :: Array{ComplexF64, 5}
    real_space_quartic_vertices :: Vector{Union{RealSpaceQuarticVerticesSUN, RealSpaceQuarticVerticesDipole}}
    real_space_cubic_vertices   :: Vector{Union{RealSpaceCubicVerticesSUN, RealSpaceCubicVerticesDipole}}
    tensormode :: Symbol
end

function calculate_real_space_quartic_vertices_sun(sys::System)
    N = sys.Ns[1]
    V41_buf = zeros(ComplexF64, N-1, N-1, N-1, N-1)
    V42_buf = zeros(ComplexF64, N-1, N-1)
    V43_buf = zeros(ComplexF64, N-1, N-1)

    real_space_quartic_vertices = RealSpaceQuarticVerticesSUN[]

    for int in sys.interactions_union
        for coupling in int.pair

            coupling.isculled && break
            V41_buf .= 0.0
            V42_buf .= 0.0
            V43_buf .= 0.0

            for (A, B) in coupling.general.data
                for σ1 in 1:N-1, σ3 in 1:N-1
                    V42_buf[σ1, σ3] += -0.5 * A[N, σ1] * B[N, σ3]
                    V43_buf[σ1, σ3] += -0.5 * A[σ1, N] * B[N, σ3]
                    for σ2 in 1:N-1, σ4 in 1:N-1
                        V41_buf[σ1, σ2, σ3, σ4] += (A[σ1, σ2] - δ(σ1, σ2)*A[N, N]) * (B[σ3, σ4] - δ(σ3, σ4)*B[N, N])
                    end
                end
            end

            quartic_vertices = RealSpaceQuarticVerticesSUN(copy(V41_buf), copy(V42_buf), copy(V43_buf))
            push!(real_space_quartic_vertices, quartic_vertices)
        end
    end

    return real_space_quartic_vertices
end

function calculate_real_space_quartic_vertices_dipole(sys::System)
    real_space_quartic_vertices = RealSpaceQuarticVerticesDipole[]

    for int in sys.interactions_union
        for coupling in int.pair
            (; isculled, bilin, biquad, general) = coupling

            isculled && break
            J = Mat3(bilin*I)
            V41 = J[3, 3]
            V42 = 1/8 * (-J[1, 1] + J[2, 2] + 1im*J[1, 2] + 1im*J[2, 1])
            V43 = 1/8 * (-J[1, 1] - J[2, 2] - 1im*J[1, 2] + 1im*J[2, 1])
            quartic_vertices = RealSpaceQuarticVerticesDipole(V41, V42, V43)
            push!(real_space_quartic_vertices, quartic_vertices)

            @assert iszero(biquad) "Biquadratic interactions not supported in :dipole_large_S for the non-perburbative calculation."
            @assert isempty(general.data)
        end
    end

    return real_space_quartic_vertices
end

function calculate_real_space_cubic_vertices_sun(sys::System)
    N = sys.Ns[1]
    V31_p_buf = zeros(ComplexF64, N-1)
    V31_m_buf = zeros(ComplexF64, N-1)
    V32_p_buf = zeros(ComplexF64, N-1, N-1, N-1)
    V32_m_buf = zeros(ComplexF64, N-1, N-1, N-1)

    real_space_cubic_vertices = RealSpaceCubicVerticesSUN[]

    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            V31_p_buf .= 0.0
            V31_m_buf .= 0.0
            V32_p_buf .= 0.0
            V32_m_buf .= 0.0

            for (A, B) in coupling.general.data
                for σ1 in 1:N-1
                    V31_p_buf[σ1] += -0.5 * A[N, N] * B[N, σ1]
                    V31_m_buf[σ1] += -0.5 * B[N, N] * A[N, σ1]
                    for σ2 in 1:N-1, σ3 in 1:N-1
                        V32_p_buf[σ1, σ2, σ3] += A[N, σ1] * (B[σ2, σ3] - δ(σ2, σ3)*B[N, N])
                        V32_m_buf[σ1, σ2, σ3] += B[N, σ1] * (A[σ2, σ3] - δ(σ2, σ3)*A[N, N])
                    end
                end
            end

            cubic_vertices = RealSpaceCubicVerticesSUN(copy(V31_p_buf), copy(V31_m_buf), copy(V32_p_buf), copy(V32_m_buf))
            push!(real_space_cubic_vertices, cubic_vertices)
        end
    end

    return real_space_cubic_vertices
end

function calculate_real_space_cubic_vertices_dipole(sys::System)
    N = sys.Ns[1]
    S = (N-1)/2
    real_space_cubic_vertices = RealSpaceCubicVerticesDipole[]

    for int in sys.interactions_union
        for coupling in int.pair
            (; isculled, bilin, biquad, general) = coupling

            isculled && break
            J = Mat3(bilin*I)
            V31 = √(S/2) * (-J[1, 3] + 1im*J[2, 3] )
            V32 = √(S/2) * (-J[3, 1] + 1im*J[3, 2] )

            cubic_vertices = RealSpaceCubicVerticesDipole(V31, V32)
            push!(real_space_cubic_vertices, cubic_vertices)

            @assert iszero(biquad) "Biquadratic interactions not supported in :dipole_large_S for the non-perburbative calculation."
            @assert isempty(general.data)
        end
    end

    return real_space_cubic_vertices
end

function NonPerturbativeTheory(swt::SpinWaveTheory, clustersize::NTuple{3, Int}; tensormode::Symbol=:loop)
    (; sys) = swt
    @assert sys.mode in (:SUN, :dipole) "Non-perturbative calculation is only supported in :SUN or :dipole mode."
    Nu1, Nu2, Nu3 = clustersize
    @assert isodd(Nu1) && isodd(Nu2) && isodd(Nu3) "Each linear dimension of the non-perturbative cluster must be odd to guarantee an equal number of two particle states for all `qcom`s."
    @assert tensormode in (:loop, :tensor) "Only `:loop` and `:tensor` are supported for the `tensormode` argument."

    L = nbands(swt)

    qs = [Vec3([i/Nu1, j/Nu2, k/Nu3]) for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]

    Es = zeros(L, Nu1, Nu2, Nu3)
    Vps = zeros(ComplexF64, 2L, 2L, Nu1, Nu2, Nu3)
    H_buf = zeros(ComplexF64, 2L, 2L)
    V_buf = zeros(ComplexF64, 2L, 2L)

    for iq in CartesianIndices(qs)
        q = qs[iq]
        dynamical_matrix!(H_buf, swt, q)
        E = bogoliubov!(V_buf, H_buf)
        Es[:, iq] = E[1:L]
        Vps[:, :, iq] = copy(V_buf)
    end

    if sys.mode == :SUN
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_sun(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_sun(sys)
    else
        (tensormode == :tensor) && (@warn "The `tensormode` argument `:tensor` is ignored in the :dipole mode.")
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(sys)
        real_space_cubic_vertices   = calculate_real_space_cubic_vertices_dipole(sys)
    end

    return NonPerturbativeTheory(swt, clustersize, qs, Es, Vps, real_space_quartic_vertices, real_space_cubic_vertices, tensormode)

end

"""
    generate_two_particle_states(clustersize, q_index::CartesianIndex{3})

For a given `q_index::CartesianIndex{3}`, this block constructs a `Dict` that 
contains all possible two-particle states sharing the same center-of-mass momentum.
The dictionary keys consist of:
 - The Cartesian indices `q1` and `q2`, representing the momenta of the two particles.
 - The band indices of the two particles.
 
The dictionary values consist of a tuple with:
 1. The center-of-mass index of the state (an integer).
 2. The bosonic symmetry factor, which is either 1 (for distinguishable particles) or 1/√2 (for identical bosons in the same state).
 3. The final two components are the global indices of the two-particle state.
"""
function generate_two_particle_states(clustersize, L::Int, q_index::CartesianIndex{3})
    Nu1, Nu2, Nu3 = clustersize
    dict_states = Dict{Tuple{CartesianIndex{3}, CartesianIndex{3}, Int, Int}, Tuple{Int, Float64, Int, Int}}()
    cartes_indices = CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    com_index = 0
    for k_index in CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3))
        qmk_index = CartesianIndex(mod(q_index[1]-k_index[1], Nu1)+1, mod(q_index[2]-k_index[2], Nu2)+1, mod(q_index[3]-k_index[3], Nu3)+1)
        for band1 in 1:L
            ci = CartesianIndex(Tuple(k_index)..., band1)
            i  = linear_indices[ci]
            for band2 in 1:L
                cj = CartesianIndex(Tuple(qmk_index)..., band2)
                j  = linear_indices[cj]
                if i ≤ j
                    com_index += 1
                    dict_states[(k_index, qmk_index, band1, band2)] = (com_index, i == j ? 1/√2 : 1.0, i, j)
                end
            end
        end
    end
    return dict_states
end


"""
    truncated_hilbert_space_dim(npt::NonPerturbativeTheory)

Calculate the dimension of the truncated Hilbert space used in the non-perturbative calculation.
"""
function truncated_hilbert_space_dim(npt::NonPerturbativeTheory)
    (; clustersize, swt) = npt
    num_1ps = nbands(swt)
    # @info "Number of single particle states per center-of-mass momentum is" num_1ps
    Nu = prod(clustersize)
    num_2ps = Int(binomial(num_1ps*Nu+2-1, 2) / Nu)
    # @info "Number of two particle states per center-of-mass momentum is" num_2ps
    dim = num_1ps + num_2ps
    return dim, num_1ps, num_2ps
end

struct ReshapedQResult
    q_reshaped_closest :: Vec3
    q_index :: CartesianIndex{3}
end

# Given q_reshaped in reciprocal lattice units (RLU) for the possibly-reshaped crystal, return a
# q in RLU for the original crystal.
function to_original_rlu(sys::System{N}, q_reshaped) where N
    return orig_crystal(sys).recipvecs \ (sys.crystal.recipvecs * q_reshaped)
end

# Given `q` in reciprocal lattice units (RLU) for the original crystal and the `npt` object, return to the `ReshapedQResult` which contains:
# - `q_reshaped_closest`: the reshaped momentum in RLU that is closest to the input momentum
# - `q_index`: the Cartesian index of the closest momentum in the non-perturbative grid
function to_reshaped_q_npt(npt::NonPerturbativeTheory, q)
    (; qs) = npt
    # Here we mod one. This is because the q_reshaped is in the reciprocal lattice unit, and we need to find the closest q in the grid.
    q_reshaped = to_reshaped_rlu(npt.swt.sys, q)
    for i in 1:3
        (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
    end
    # Fold the reshaped wave vector within in first magnetic Brillouin zone
    q_reshaped_folded = mod.(q_reshaped, 1.0)
    G_mag = q_reshaped - q_reshaped_folded
    for i in 1:3
        (abs(q_reshaped_folded[i]) < 1e-12) && (q_reshaped_folded = setindex(q_reshaped_folded, 0.0, i))
    end
    norm_diff, q_index = findmin(x -> norm(x - q_reshaped_folded), qs)

    q_reshaped_closest = qs[q_index] + G_mag

    if norm_diff > 1e-12
        q_closest = to_original_rlu(npt.swt.sys, q_reshaped_closest)
        Δq = norm(orig_crystal(npt.swt.sys).recipvecs * (q - q_closest))
        @warn "The requested momentum $q is not available in the set of `qs` used for the NPT calculation. The closest available momentum $q_closest is used instead (‖Δq‖ = $Δq)."
    end

    return ReshapedQResult(q_reshaped_closest, q_index)
end

function q_space_path_npt(npt::NonPerturbativeTheory, qs; labels=nothing)
    (; clustersize) = npt

    reshaped_q_res = [to_reshaped_q_npt(npt, q) for q in qs]
    length_qs = length(qs)

    path = Vec3[]
    markers = Int[]

    for i in 1:length_qs - 1
        push!(markers, length(path)+1)
        q_reshaped_s = reshaped_q_res[i].q_reshaped_closest
        q_reshaped_e = reshaped_q_res[i+1].q_reshaped_closest
        Δq_reshaped = q_reshaped_e - q_reshaped_s
        Δns = round.(Int, abs.(Δq_reshaped .* collect(clustersize)))
        Δn = gcd(gcd(Δns[1], Δns[2]), Δns[3])
        j_end = i == length_qs - 1 ? Δn : Δn - 1
        for j in 0:j_end
            q_reshaped = q_reshaped_s + j/Δn * Δq_reshaped
            q = to_original_rlu(npt.swt.sys, q_reshaped)
            push!(path, q)
        end
    end

    push!(markers, length(path))

    labels = @something labels vec3_to_string.(qs)
    xticks = (markers, labels)
    return QPath(path, xticks)
end

"""
    q_space_polygon_npt(npt::NonPerturbativeTheory, qs; plane_axis=3, atol=1e-12)

Construct a set of NPT-compatible q-points inside a closed polygonal region.

The input `qs` are polygon vertices in the reciprocal lattice units of the
original crystal. The first and last vertices must close the polygon. Each
vertex is first snapped to the nearest available NPT momentum using
`to_reshaped_q_npt`.

The polygon-inclusion test is performed in reshaped reciprocal lattice units.
By default, `plane_axis=3`, so the polygon is assumed to live in the first two
reshaped coordinates, with the third coordinate fixed.

Returns a `QPoints` object whose `qs` are expressed in the reciprocal lattice
units of the original crystal.
"""
function q_space_polygon_npt(npt::NonPerturbativeTheory, qs; plane_axis::Int=3, atol::Float64=1e-12)
    @assert 1 ≤ plane_axis ≤ 3 "`plane_axis` must be 1, 2, or 3."
    @assert length(qs) ≥ 4 "A closed polygon requires at least four vertices, including the repeated final vertex."

    (; clustersize) = npt
    sys = npt.swt.sys

    # Snap the user-specified polygon vertices to the NPT momentum grid.
    #
    # The input vertices are arbitrary q-points in the original crystal RLU.
    # `to_reshaped_q_npt` converts each one to reshaped RLU, folds it into the
    # first reshaped reciprocal unit cell, snaps it to the nearest finite-size
    # momentum in `npt.qs`, and then restores the corresponding magnetic
    # reciprocal-lattice image.
    reshaped_q_res = [to_reshaped_q_npt(npt, q) for q in qs]
    vertices_closed = [res.q_reshaped_closest for res in reshaped_q_res]

    # Check that the polygon is closed after NPT snapping.
    #
    # This is the relevant closure condition because all later geometric tests
    # are performed using the snapped vertices in reshaped RLU.
    if norm(vertices_closed[1] - vertices_closed[end]) > atol
        q_start = to_original_rlu(sys, vertices_closed[1])
        q_end = to_original_rlu(sys, vertices_closed[end])
        error("The polygon is not closed after snapping to the NPT grid. " *
              "The snapped first point is $q_start, while the snapped last point is $q_end.")
    end

    # Remove the repeated final vertex. The polygon-inclusion helper expects
    # each distinct vertex only once; it closes the final edge internally.
    vertices = vertices_closed[1:end-1]

    # The polygon is assumed to lie in a coordinate plane of the reshaped
    # reciprocal coordinates. For example, `plane_axis = 3` means the polygon
    # lies in the q1-q2 plane with fixed q3.
    plane_coord = vertices[1][plane_axis]
    for v in vertices
        if abs(v[plane_axis] - plane_coord) > atol
            error("The snapped polygon vertices are not coplanar with fixed coordinate " *
                  "`plane_axis = $plane_axis`. Got coordinates " *
                  "$(getindex.(vertices, plane_axis)).")
        end
    end

    # These are the two coordinate axes used for the 2D polygon test.
    # For `plane_axis = 3`, this gives axes2 = (1, 2).
    axes2 = Tuple(i for i in 1:3 if i != plane_axis)

    # Convert the 3D reshaped-RLU vertices into 2D coordinates in the polygon
    # plane. The helper functions below only solve a 2D geometry problem.
    polygon2 = [(v[axes2[1]], v[axes2[2]]) for v in vertices]

    # Determine the magnetic reciprocal-lattice images that can overlap the
    # polygon bounding box. This matters because the snapped polygon vertices
    # may lie in an extended-zone image, while `npt.qs` itself stores only the
    # folded representatives in [0,1)^3.
    mins = ntuple(a -> minimum(v[a] for v in vertices), 3)
    maxs = ntuple(a -> maximum(v[a] for v in vertices), 3)

    G_ranges = ntuple(a -> (floor(Int, mins[a]) - 1):(ceil(Int, maxs[a]) + 1), 3)

    qs_out = Vec3[]

    for iq in CartesianIndices(npt.qs)
        q_base = npt.qs[iq]

        for n1 in G_ranges[1], n2 in G_ranges[2], n3 in G_ranges[3]
            G = Vec3([n1, n2, n3])
            q_reshaped = q_base + G

            # Restrict to the same coordinate plane as the polygon.
            if abs(q_reshaped[plane_axis] - plane_coord) > atol
                continue
            end

            # Project this candidate q-point to the 2D polygon coordinates.
            p = (q_reshaped[axes2[1]], q_reshaped[axes2[2]])

            # Keep the point if it lies inside the polygon or exactly on its
            # boundary. Boundary inclusion is intentional: a q-point on a
            # polygon edge should be part of the selected polygonal region.
            if _point_in_polygon_or_on_boundary(p, polygon2; atol=atol)
                q = to_original_rlu(sys, q_reshaped)
                push!(qs_out, q)
            end
        end
    end

    # Deterministic ordering. This is not a path order; it is just a stable
    # lexicographic ordering of the selected q-points in original RLU.
    sort!(qs_out, by = q -> (q[1], q[2], q[3]))

    return QPoints(qs_out)
end


function _point_in_polygon_or_on_boundary(p, polygon; atol::Float64=1e-12)
    n = length(polygon)
    @assert n ≥ 3 "A polygon requires at least three distinct vertices."

    # This helper answers the purely geometric question:
    #
    #     Is the 2D point `p` inside the polygon, including its boundary?
    #
    # It does not know anything about Sunny, reciprocal lattices, or the NPT
    # grid. By the time this function is called, both `p` and `polygon` have
    # already been projected to ordinary 2D coordinates in the chosen
    # reshaped-RLU plane.

    # First check whether `p` lies exactly on any polygon edge.
    #
    # The usual ray-crossing algorithm is meant for deciding strictly inside
    # versus outside. Boundary cases are delicate: a point exactly on an edge
    # or exactly at a vertex can be counted inconsistently depending on the
    # direction of the ray and floating-point roundoff.
    #
    # For the q-grid selection problem, we want the convention
    #
    #     polygonal region = interior + boundary.
    #
    # Therefore boundary points are accepted explicitly before running the
    # inside/outside test.
    for i in 1:n
        a = polygon[i]
        b = polygon[i == n ? 1 : i + 1]
        if _point_on_segment(p, a, b; atol=atol)
            return true
        end
    end

    # Ray-crossing test for the remaining non-boundary points.
    #
    # Imagine drawing a horizontal ray starting from `p` and going to the
    # right. Count how many times this ray crosses the polygon boundary.
    #
    #     odd number of crossings  -> inside
    #     even number of crossings -> outside
    #
    # Instead of explicitly counting crossings, we flip the Boolean `inside`
    # each time the ray crosses an edge.
    inside = false
    x, y = p

    j = n
    for i in 1:n
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # The horizontal ray at height `y` can cross this edge only if the two
        # endpoints lie on opposite sides of the horizontal line through `p`.
        #
        # The strict comparisons avoid double-counting vertices. Boundary
        # cases have already been handled above.
        crosses = ((yi > y) != (yj > y))

        if crosses
            # Compute the x-coordinate where this polygon edge intersects the
            # horizontal line passing through `p`.
            x_intersect = xi + (y - yi) * (xj - xi) / (yj - yi)

            # If the intersection lies to the right of `p`, the ray crosses
            # this edge, so flip inside/outside.
            if x < x_intersect
                inside = !inside
            end
        end

        j = i
    end

    return inside
end


function _point_on_segment(p, a, b; atol::Float64=1e-12)
    x, y = p
    x1, y1 = a
    x2, y2 = b

    # This helper checks whether the 2D point `p` lies on the finite line
    # segment from endpoint `a` to endpoint `b`.
    #
    # It performs two tests:
    #
    #   1. Collinearity:
    #        `p`, `a`, and `b` must lie on the same infinite line.
    #
    #   2. Between-endpoints:
    #        once collinearity is satisfied, `p` must lie between `a` and `b`,
    #        not beyond either endpoint.
    #
    # The tolerance `atol` is used because the q-points and snapped polygon
    # vertices are floating-point numbers.

    dx = x2 - x1
    dy = y2 - y1

    # Test 1: collinearity.
    #
    # The 2D cross product between the vectors
    #
    #     a -> p  = (x - x1, y - y1)
    #     a -> b  = (dx, dy)
    #
    # should vanish if the two vectors are parallel. A nonzero value means `p`
    # is not on the same infinite line as the segment.
    cross = (x - x1) * dy - (y - y1) * dx
    if abs(cross) > atol
        return false
    end

    # Test 2: between-endpoints.
    #
    # The dot product with the segment direction tells us where `p` sits along
    # the line from `a` to `b`.
    #
    #     dotprod < 0      -> p is behind a
    #     dotprod > len2   -> p is beyond b
    #     otherwise        -> p lies between a and b
    dotprod = (x - x1) * dx + (y - y1) * dy
    if dotprod < -atol
        return false
    end

    len2 = dx^2 + dy^2
    if dotprod - len2 > atol
        return false
    end

    return true
end