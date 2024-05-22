function two_particle_hamiltonian!(H::Matrix{ComplexF64}, npt::NonPerturbativeTheory, qcom_carts_index::CartesianIndex)
    H .= 0.0
    (; two_particle_states, swt, clustersize) = npt
    Nu = clustersize[1] * clustersize[2] * clustersize[3]
    L = nbands(swt)

    com_states = two_particle_states[qcom_carts_index]

    q_pairs = Tuple{Vec3, Vec3, CartesianIndex, CartesianIndex}[]
    for state in com_states
        push!(q_pairs, (state.q1, state.q2, state.q1_carts_index, state.q2_carts_index))
    end

    unique!(q_pairs)

    dict = Dict{Tuple{Vec3, Vec3, CartesianIndex, CartesianIndex}, Int}()
    for (i, q_pair) in enumerate(q_pairs)
        dict[q_pair] = i
    end

    num_qpairs = length(q_pairs)
    ret = zeros(ComplexF64, L, L, L, L, num_qpairs, num_qpairs);

    for q_pair1 in q_pairs, q_pair2 in q_pairs
        i = dict[q_pair1]
        j = dict[q_pair2]

        if swt.sys.mode == :SUN
            view(ret, :, :, :, :, i, j) .= quartic_vertex_SUN(npt, [-q_pair1[1], -q_pair1[2], q_pair2[1], q_pair2[2]], [q_pair1[3], q_pair1[4], q_pair2[3], q_pair2[4]])
        else
            view(ret, :, :, :, :, i, j) .= quartic_vertex_dipole(npt, [-q_pair1[1], -q_pair1[2], q_pair2[1], q_pair2[2]], [q_pair1[3], q_pair1[4], q_pair2[3], q_pair2[4]])
        end

    end

    for state_i in com_states, state_j in com_states
        i = state_i.global_index_i
        j = state_i.global_index_j
        k = state_j.global_index_i
        l = state_j.global_index_j

        ζij = state_i.ζ
        ζkl = state_j.ζ

        com_i = state_i.com_index
        com_j = state_j.com_index

        band1 = state_i.band1
        band2 = state_i.band2
        band3 = state_j.band1
        band4 = state_j.band2

        tuple_i = (state_i.q1, state_i.q2, state_i.q1_carts_index, state_i.q2_carts_index)
        tuple_j = (state_j.q1, state_j.q2, state_j.q1_carts_index, state_j.q2_carts_index)

        # The quartic vertex is multiplied by a factor of `Nm` (number of magnetic sites) in 
        # the magnetic unit cell. This is because the final results are reported in the Brillouin zone of the chemical unit cell.
        pairkey_i = dict[tuple_i]
        pairkey_j = dict[tuple_j]
        H[com_i, com_j] += (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) * (npt.Es[band1, state_i.q1_carts_index] + npt.Es[band2, state_i.q2_carts_index]) * ζij^2 + ret[band1, band2, band3, band4, pairkey_i, pairkey_j] * ζij * ζkl / Nu
    end
end


# TODO: integrate this with Lanczos
# function two_particle_dispersion(npt::NonPerturbativeTheory, qs, niters)
#     (; clustersize, swt) = npt
#     Nu1, Nu2, Nu3 = clustersize
#     # number of two-particle states
#     L2p = npt.two_particle_states[1, 1, 1]
#     H2p = zeros(ComplexF64, L2p, L2p)
#     disp_2p = zeros(Float64, niters, length(qs))

#     qmags = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]

#     com_indices = CartesianIndex[]

#     # Find the nearest center of mass index in the magnetic Brillouin zone
#     for q in qs
#         q_reshaped = to_reshaped_rlu(swt.sys, q)
#         qcom_carts_index = findmin(x -> norm(x - q_reshaped), qmags)[2]
#         push!(com_indices, qcom_carts_index)
#     end

#     for (i, qcom_carts_index) in enumerate(com_indices)
#         two_particle_hamiltonian!(H2p, npt, qcom_carts_index)
#         # TODO: Add a check for thehermitianity of the matrix
#         hermitianpart!(H2p)
#         H_tridiag = lanczos(H2p, niters)
#         view(disp_2p, :, i), _ = eigen(H_tridiag) 
#     end

#     return reshape_dispersions(disp_2p)
# end