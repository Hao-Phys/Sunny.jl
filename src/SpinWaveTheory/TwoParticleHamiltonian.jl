function calculate_quartic_vertices(npt::NonPerturbativeTheory)
    (; swt, clustersize) = npt
    sys = swt.sys
    L = nbands(swt)
    Nu1, Nu2, Nu3 = clustersize
    qs = [Vec3([i/Nu1, j/Nu2, k/Nu3]) for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    cartes_indices = CartesianIndices((1:Nu1, 1:Nu2, 1:Nu3))
    linear_indices = LinearIndices(cartes_indices)

    numqs = Nu1*Nu2*Nu3

    ret = zeros(ComplexF64, L, L, L, L, numqs, numqs, numqs, numqs)

    for ci in cartes_indices, cj in cartes_indices, ck in cartes_indices, cl in cartes_indices
        i = linear_indices[ci]
        j = linear_indices[cj]
        k = linear_indices[ck]
        l = linear_indices[cl]

        if sys.mode == :SUN
            view(ret, :, :, :, :, i, j, k, l) .= quartic_vertex_SUN(npt, [-qs[ci], -qs[cj], qs[ck], qs[cl]], [ci, cj, ck, cl])
        else

        end
    end

    return ret
end

function two_particle_hamiltonian!(H::Matrix{ComplexF64}, npt::NonPerturbativeTheory, quartic_vertices::Array{ComplexF64, 8}, qcom_carts_index)
    H .= 0.0
    (; two_particle_states) = npt
    qcom_carts_index = CartesianIndex(qcom_carts_index)

    com_states = two_particle_states[qcom_carts_index]

    for state_i in com_states, state_j in com_states
        q1_index = state_i.q1_linear_index
        q2_index = state_i.q2_linear_index
        q3_index = state_j.q1_linear_index
        q4_index = state_j.q2_linear_index
        q1_carts_index = state_i.q1_carts_index
        q2_carts_index = state_i.q2_carts_index

        i = state_i.global_i
        j = state_i.global_j

        k = state_j.global_i
        l = state_j.global_j

        ζij = state_i.ζ
        ζkl = state_j.ζ

        com_i = state_i.com_index
        com_j = state_j.com_index

        band1 = state_i.band1
        band2 = state_i.band2
        band3 = state_j.band1
        band4 = state_j.band2

        # Note that in the definition of the quartic vertex, we've already normalized the result by number of unit cells
        H[com_i, com_j] += δ(i, k) * δ(j, l) * (npt.Es[band1, q1_carts_index] + npt.Es[band2, q2_carts_index]) * ζij^2 + quartic_vertices[band1, band2, band3, band4, q1_index, q2_index, q3_index, q4_index] * ζij * ζkl
    end

end