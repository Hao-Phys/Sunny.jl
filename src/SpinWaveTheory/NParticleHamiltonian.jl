function one_particle_hamiltonian!(H, npt::NonPerturbativeTheory, q_index::CartesianIndex{3}; single_particle_correction::Bool=true, opts...)
    H .= 0.0
    L  = nbands(npt.swt)
    Es = npt.Es[:, q_index]
    q  = npt.qs[q_index]

    # Diagonal part from the non-interacting theory
    for i in 1:L
        H[i, i] += Es[i]
    end

    # Diagonal and off-diagonal part from normal-ordering
    if single_particle_correction
        if npt.swt.sys.mode == :SUN
            Q2 = quadratic_vertex_SUN(npt, q, q_index; opts...)
        else
            Q2 = quadratic_vertex_dipole(npt, q, q_index; opts...)
        end

        for i in 1:L, j in 1:L
            H[i, j] += Q2[i, j]
        end
    end
end

function two_particle_hamiltonian!(H, npt::NonPerturbativeTheory, q_index::CartesianIndex{3})
    H .= 0.0
    (; swt, clustersize) = npt
    qs = npt.qs
    q  = qs[q_index]
    Nu1, Nu2, Nu3 = clustersize
    Nu = Nu1 * Nu2 * Nu3
    L = nbands(swt)

    dict_states = generate_two_particle_states(clustersize, L, q_index)

    # Quartic vertex function
    quartic_vertex_fun = swt.sys.mode == :SUN ? quartic_vertex_SUN : quartic_vertex_dipole

    # Quartic vertex buffer
    U4 = zeros(ComplexF64, L, L, L, L)
    for k1_index in CartesianIndices(qs)
        k1 = qs[k1_index]
        qmk1 = mod.(q - k1, 1.0)
        qmk1_index = CartesianIndex(mod(q_index[1]-k1_index[1], Nu1)+1, mod(q_index[2]-k1_index[2], Nu2)+1, mod(q_index[3]-k1_index[3], Nu3)+1)
        for k2_index in CartesianIndices(qs)
            k2 = qs[k2_index]
            qmk2 = mod.(q - k2, 1.0)
            qmk2_index = CartesianIndex(mod(q_index[1]-k2_index[1], Nu1)+1, mod(q_index[2]-k2_index[2], Nu2)+1, mod(q_index[3]-k2_index[3], Nu3)+1)
            U4 .= quartic_vertex_fun(npt, (-k1, -qmk1, k2, qmk2), (k1_index, qmk1_index, k2_index, qmk2_index))
            for band1 in 1:L, band2 in 1:L, band3 in 1:L, band4 in 1:L
                if haskey(dict_states, (k1_index, qmk1_index, band1, band2)) && haskey(dict_states, (k2_index, qmk2_index, band3, band4))
                    (com_1, ζij, i, j) = dict_states[(k1_index, qmk1_index, band1, band2)]
                    (com_2, ζkl, k, l) = dict_states[(k2_index, qmk2_index, band3, band4)]
                    H[com_1, com_2] += (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) * (npt.Es[band1, k1_index] + npt.Es[band2, qmk1_index]) * ζij^2 + U4[band1, band2, band3, band4] * ζij * ζkl / Nu
                end
            end
        end
    end

end

function one_to_two_particle_hamiltonian!(H, npt::NonPerturbativeTheory, q_index::CartesianIndex{3})
    H .= 0.0
    (; swt, clustersize) = npt
    qs = npt.qs
    q  = qs[q_index]
    Nu1, Nu2, Nu3 = clustersize
    Nu = Nu1 * Nu2 * Nu3
    L = nbands(swt)

    dict_states = generate_two_particle_states(clustersize, L, q_index)

    cubic_vertex_fun = swt.sys.mode == :SUN ? cubic_vertex_SUN : cubic_vertex_dipole

    U3 = zeros(ComplexF64, L, L, L)
    for k_index in CartesianIndices(qs)
        k = qs[k_index]
        qmk = mod.(q - k, 1.0)
        qmk_index = CartesianIndex(mod(q_index[1]-k_index[1], Nu1)+1, mod(q_index[2]-k_index[2], Nu2)+1, mod(q_index[3]-k_index[3], Nu3)+1)
        U3 .= cubic_vertex_fun(npt, (-q, k, qmk), (q_index, k_index, qmk_index))
        for band in 1:L, band1 in 1:L, band2 in 1:L
            if haskey(dict_states, (k_index, qmk_index, band1, band2))
                (com, ζjk, _, _) = dict_states[(k_index, qmk_index, band1, band2)]
                H[band, com] += U3[band, band1, band2] * ζjk / √Nu
            end
        end
    end
end

function vacuum_to_two_particle_hamiltonian!(H, npt::NonPerturbativeTheory)
    H .= 0.0

    (; swt, clustersize, qs) = npt
    Nu1, Nu2, Nu3 = clustersize
    Nu = Nu1 * Nu2 * Nu3
    L = nbands(swt)

    # Two-particle states with center-of-mass momentum equal to (0, 0, 0)
    dict_states = generate_two_particle_states(clustersize, L, CartesianIndex((1, 1, 1)))

    vacuum_vertex_fun = swt.sys.mode == :SUN ? vacuum_V4_SUN : vacuum_V4_dipole

    V4 = zeros(ComplexF64, L, L)

    for kp_index in CartesianIndices(qs)
        kp = qs[kp_index]
        km = mod.(-kp, 1.0)
        km_index = CartesianIndex(mod(1-kp_index[1], Nu1)+1, mod(1-kp_index[2], Nu2)+1, mod(1-kp_index[3], Nu3)+1)

        V4 .= vacuum_vertex_fun(npt, (kp, km), (kp_index, km_index))

        for band1 in 1:L, band2 in 1:L
            if haskey(dict_states, (kp_index, km_index, band1, band2))
                (com, ζjk, _, _) = dict_states[(kp_index, km_index, band1, band2)]
                H[com] += V4[band1, band2] * ζjk / Nu
            end
        end
    end

end

function truncated_hilbert_space_eigen(npt::NonPerturbativeTheory, q; single_particle_correction::Bool=true, opts...)
    (; clustersize, swt) = npt
    Nu1, Nu2, Nu3 = clustersize
    Nu = Nu1 * Nu2 * Nu3

    q_index = to_reshaped_q_npt(npt, q).q_index

    num_1ps = nbands(swt)
    # Number of two-particle states is given by the following combinatorial formula:
    num_2ps = Int(binomial(Nu*num_1ps+2-1, 2) / Nu)

    H = zeros(ComplexF64, num_1ps+num_2ps, num_1ps+num_2ps)
    H1ps = view(H, 1:num_1ps, 1:num_1ps)
    H2ps = view(H, num_1ps+1:num_1ps+num_2ps, num_1ps+1:num_1ps+num_2ps)
    H12ps = view(H, 1:num_1ps, num_1ps+1:num_1ps+num_2ps)
    H21ps = view(H, num_1ps+1:num_1ps+num_2ps, 1:num_1ps)
    one_particle_hamiltonian!(H1ps, npt, q_index; single_particle_correction, opts...)
    two_particle_hamiltonian!(H2ps, npt, q_index)
    one_to_two_particle_hamiltonian!(H12ps, npt, q_index)
    @. H21ps = copy(H12ps')

    hermitianpart!(H)
    E, V = eigen!(H; sortby = x -> -1/real(x))

    return E, V
end