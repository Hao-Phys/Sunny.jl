struct TwoParticleState
    q1 :: Vec3
    q2 :: Vec3
    qcom :: Vec3
    q1_carts_index :: CartesianIndex
    q2_carts_index :: CartesianIndex
    qcom_carts_index :: CartesianIndex
    band1 :: Int
    band2 :: Int
    global_index_i :: Int
    global_index_j :: Int
    com_index :: Int
    ζ :: Float64
end

function generate_two_particle_basis(cluster_size::NTuple{3, Int}, numbands::Int)
    Nu1, Nu2, Nu3 = cluster_size
    qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    cartes_indices = CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3, 1:numbands))
    linear_indices = LinearIndices(cartes_indices)

    tp_counts = zeros(Int, Nu1, Nu2, Nu3)
    tp_states = [TwoParticleState[] for _ in 1:Nu1, _ in 1:Nu2, _ in 1:Nu3]

    for ci in cartes_indices, cj in cartes_indices
        i = linear_indices[ci]
        j = linear_indices[cj]
        if i ≤ j
            q1_carts_index = CartesianIndex(ci[1], ci[2], ci[3])
            q2_carts_index = CartesianIndex(cj[1], cj[2], cj[3])
            q1 = qs[q1_carts_index]
            q2 = qs[q2_carts_index]
            qcom = mod.(q1+q2, 1.0)
            qcom_carts_index = findfirst(x -> x ≈ qcom, qs)

            tp_counts[qcom_carts_index] += 1
            ζ = i == j ? 1/√2 : 1.0
            tp_state = TwoParticleState(Vec3(q1), Vec3(q2), Vec3(qcom), q1_carts_index, q2_carts_index, qcom_carts_index, ci[4], cj[4], i, j, tp_counts[qcom_carts_index], ζ)
            push!(tp_states[qcom_carts_index], tp_state)
        end
    end

    return tp_states

end

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

struct NonPerturbativeTheory
    swt :: SpinWaveTheory
    clustersize :: NTuple{3, Int}   # Cluster size for number of magnetic unit cell
    two_particle_states :: Array{Vector{TwoParticleState}, 3}
    Es :: Array{Float64, 4}
    Vps :: Array{ComplexF64, 5}
    real_space_quartic_vertices :: Vector{Union{RealSpaceQuarticVerticesSUN, RealSpaceQuarticVerticesDipole}}
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

            quartic_vertices = RealSpaceQuarticVerticesSUN(V41_buf, V42_buf, V43_buf)
            push!(real_space_quartic_vertices, quartic_vertices)
        end
    end

    return real_space_quartic_vertices
end

function calculate_real_space_quartic_vertices_dipole(sys::System)
    N = sys.Ns[1]
    S = (N-1)/2
    real_space_quartic_vertices = RealSpaceQuarticVerticesDipole[]

    for int in sys.interactions_union
        for coupling in int.pair
            (; isculled, bilin, biquad, general) = coupling

            isculled && break
            J = Mat3(bilin*I)
            V41 = J[3, 3] / S
            V42 = 1/(4S) * (-J[1, 1] + J[2, 2] + 1im*J[1, 2] + 1im*J[2, 1])
            V43 = 1/(4S) * (-J[1, 1] - J[2, 2] - 1im*J[1, 2] + 1im*J[2, 1])
            quartic_vertices = RealSpaceQuarticVerticesDipole(V41, V42, V43)
            push!(real_space_quartic_vertices, quartic_vertices)

            @assert iszero(biquad) "Biquadratic interactions not supported in :dipole_large_S for the non-perburbative calculation."
            @assert isempty(general.data)
        end
    end

    return real_space_quartic_vertices
end

function NonPerturbativeTheory(swt::SpinWaveTheory, clustersize::NTuple{3, Int})
    (; sys) = swt
    @assert sys.mode in (:SUN, :dipole, :dipole_large_S)
    Nu1, Nu2, Nu3 = clustersize
    @assert isodd(Nu1) && isodd(Nu2) && isodd(Nu3) "Each linear dimension of the non-perturbative cluster must be odd to guarantee an equal number of two particle states for all `qcom`s."

    L = nbands(swt)
    two_particle_states = generate_two_particle_basis(clustersize, L)

    qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]

    Es = zeros(L, Nu1, Nu2, Nu3)
    Vps = zeros(ComplexF64, 2L, 2L, Nu1, Nu2, Nu3)
    H_buf = zeros(ComplexF64, 2L, 2L)
    V_buf = zeros(ComplexF64, 2L, 2L)

    if sys.mode == :SUN
        for iq in CartesianIndices(qs)
            q = qs[iq]
            swt_hamiltonian_SUN!(H_buf, swt, Vec3(q))
            E = bogoliubov!(V_buf, H_buf)
            Es[:, iq] = E
            Vps[:, :, iq] = deepcopy(V_buf)
        end
    else
        for iq in CartesianIndices(qs)
            q = qs[iq]
            swt_hamiltonian_dipole!(H_buf, swt, Vec3(q))
            E = bogoliubov!(V_buf, H_buf)
            Es[:, iq] = E
            Vps[:, :, iq] = deepcopy(V_buf)
        end
    end

    if sys.mode == :SUN
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_sun(sys)
    else
        real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(sys)
    end

    return NonPerturbativeTheory(swt, clustersize, two_particle_states, Es, Vps, real_space_quartic_vertices)

end