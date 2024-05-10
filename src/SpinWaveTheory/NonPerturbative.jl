struct TwoParticleState
    q1 :: Vec3
    q2 :: Vec3
    qcom :: Vec3
    q1_index :: CartesianIndex
    q2_index :: CartesianIndex
    qcom_index :: CartesianIndex
    band1 :: Int
    band2 :: Int
    two_particle_basis_index :: Int
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
            q1_index = CartesianIndex(ci[1], ci[2], ci[3])
            q2_index = CartesianIndex(cj[1], cj[2], cj[3])
            q1 = qs[q1_index]
            q2 = qs[q2_index]
            qcom = mod.(q1+q2, 1.0)
            qcom_index = findfirst(x -> x ≈ qcom, qs)

            tp_counts[qcom_index] += 1
            ζ = i == j ? 1/√2 : 1.0
            tp_state = TwoParticleState(Vec3(q1), Vec3(q2), Vec3(qcom), q1_index, q2_index, qcom_index, ci[4], cj[4], tp_counts[qcom_index], ζ)
            push!(tp_states[qcom_index], tp_state)
        end
    end

    return tp_states

end

struct RealSpaceQuarticVertices
    V41 :: Array{ComplexF64, 4}
    V42 :: Array{ComplexF64, 2}
    V43 :: Array{ComplexF64, 2}
end

struct NonPerturbativeTheory
    swt :: SpinWaveTheory
    clustersize :: NTuple{3, Int}
    two_particle_states :: Array{Vector{TwoParticleState}, 3}
    Es :: Array{Float64, 4}
    Vps :: Array{ComplexF64, 5}
    Vms :: Array{ComplexF64, 5}
    real_space_quartic_vertices :: Vector{RealSpaceQuarticVertices}
end

function calculate_real_space_quartic_vertices(swt::SpinWaveTheory)
    sys = swt.sys
    N = sys.Ns[1]
    V41_buf = zeros(ComplexF64, N-1, N-1, N-1, N-1)
    V42_buf = zeros(ComplexF64, N-1, N-1)
    V43_buf = zeros(ComplexF64, N-1, N-1)

    real_space_quartic_vertices = RealSpaceQuarticVertices[]

    for int in sys.interactions_union
        for coupling in int.pair

            coupling.isculled && break
            V41_buf .= 0.0
            V42_buf .= 0.0
            V43_buf .= 0.0

            for (A, B) in coupling.general.data
                for σ1 in 1:N-1, σ2 in 1:N-1, σ3 in 1:N-1
                    V42_buf[σ1, σ3] += -0.5 * A[N, σ1] * B[N, σ3]
                    V43_buf[σ1, σ3] += -0.5 * A[σ1, N] * B[N, σ3]
                    for σ4 in 1:N-1
                        V41_buf[σ1, σ2, σ3, σ4] += (A[σ1, σ2] - δ(σ1, σ2)*A[N, N]) * (B[σ3, σ4] - δ(σ3, σ4)*B[N, N])
                    end
                end
            end

            quartic_vertices = RealSpaceQuarticVertices(V41_buf, V42_buf, V43_buf)
            push!(real_space_quartic_vertices, quartic_vertices)
        end
    end

    return real_space_quartic_vertices
end

function NonPerturbativeTheory(swt::SpinWaveTheory, clustersize::NTuple{3, Int})
    @assert swt.sys.mode == :SUN "The non-perturbative calculation is only supported in the :SUN mode"
    Nu1, Nu2, Nu3 = clustersize
    @assert isodd(Nu1) && isodd(Nu2) && isodd(Nu3) "Each linear dimension of the non-perturbative cluster must be odd to guarantee an equal number of two particle states for all `qcom`s."

    L = nbands(swt)
    two_particle_states = generate_two_particle_basis(clustersize, L)

    qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]

    Es = zeros(L, Nu1, Nu2, Nu3)
    Vps = zeros(ComplexF64, 2L, 2L, Nu1, Nu2, Nu3)
    Vms = zeros(ComplexF64, 2L, 2L, Nu1, Nu2, Nu3)
    H_buf = zeros(ComplexF64, 2L, 2L)
    V_buf = zeros(ComplexF64, 2L, 2L)

    for iq in CartesianIndices(qs)
        q = qs[iq]
        swt_hamiltonian_SUN!(H_buf, swt, Vec3(q))
        E = bogoliubov!(V_buf, H_buf)
        Es[:, iq] = E
        Vps[:, :, iq] = deepcopy(V_buf)
        swt_hamiltonian_SUN!(H_buf, swt, Vec3(-q))
        bogoliubov!(V_buf, H_buf)
        Vms[:, :, iq] = deepcopy(V_buf)
    end

    real_space_quartic_vertices = calculate_real_space_quartic_vertices(swt)

    return NonPerturbativeTheory(swt, clustersize, two_particle_states, Es, Vps, Vms, real_space_quartic_vertices)

end