function quadratic_Q41_dipole(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           n₁, n₂ = cart[1], cart[2]
           for m in 1:L
               f[i] += conj(Vq[α₁, n₁]) * Vk[α₂+L, m] * conj(Vk[α₃+L, m]) * Vq[α₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vq[α₂+L, n₂] * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vk[α₂+L, m] * Vq[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂, m]) * conj(Vq[α₃+L, n₁]) * Vq[α₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂, n₁]) * conj(Vk[α₃+L, m]) * Vq[α₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂, m]) * Vq[α₃, n₂] * conj(Vq[α₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * conj(Vq[α₂, n₁]) * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂, n₁]) * Vq[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂+L, m] * conj(Vk[α₃+L, m]) * conj(Vq[α₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂+L, n₂] * conj(Vk[α₃+L, m]) * conj(Vq[α₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂+L, m] * conj(Vq[α₃+L, n₁]) * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂+L, n₂] * conj(Vq[α₃+L, n₁]) * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q41_buf = reshape(ret[1], (L, L))

    return Q41_buf
end

function quadratic_Q42_dipole(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           n₁, n₂ = cart[1], cart[2]
           for m in 1:L
               f[i] += conj(Vq[α₁, n₁]) * Vk[α₂, m] * conj(Vk[α₃+L, m]) * Vq[α₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vq[α₂, n₂] * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vk[α₂, m] * Vq[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂+L, m]) * conj(Vq[α₃+L, n₁]) * Vq[α₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂+L, n₁]) * conj(Vk[α₃+L, m]) * Vq[α₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂+L, m]) * Vq[α₃, n₂] * conj(Vq[α₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * conj(Vq[α₂+L, n₁]) * Vk[α₃, m] * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂+L, n₁]) * Vq[α₃, n₂] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂, m] * conj(Vk[α₃+L, m]) * conj(Vq[α₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂, n₂] * conj(Vk[α₃+L, m]) * conj(Vq[α₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂, m] * conj(Vq[α₃+L, n₁]) * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂, n₂] * conj(Vq[α₃+L, n₁]) * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q42_buf = reshape(ret[1], (L, L))

    return Q42_buf
end


function quadratic_Q43_dipole(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           n₁, n₂ = cart[1], cart[2]
           for m in 1:L
               f[i] += conj(Vq[α₁, n₁]) * Vk[α₂+L, m] * conj(Vk[α₃, m]) * Vq[α₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vq[α₂+L, n₂] * Vk[α₃+L, m] * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[α₁, n₁]) * Vk[α₂+L, m] * Vq[α₃+L, n₂] * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂, m]) * conj(Vq[α₃, n₁]) * Vq[α₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂, n₁]) * conj(Vk[α₃, m]) * Vq[α₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vk[α₂, m]) * Vq[α₃+L, n₂] * conj(Vq[α₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * conj(Vq[α₂, n₁]) * Vk[α₃+L, m] * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[α₁+L, m] * conj(Vq[α₂, n₁]) * Vq[α₃+L, n₂] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂+L, m] * conj(Vk[α₃, m]) * conj(Vq[α₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂+L, n₂] * conj(Vk[α₃, m]) * conj(Vq[α₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[α₁+L, n₂] * Vk[α₂+L, m] * conj(Vq[α₃, n₁]) * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[α₁+L, m] * Vq[α₂+L, n₂] * conj(Vq[α₃, n₁]) * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q43_buf = reshape(ret[1], (L, L))

    return Q43_buf
end

function quadratic_vertex_dipole(npt::NonPerturbativeTheory, q::Vec3, q_index::CartesianIndex{3}; opts...)
    (; swt, real_space_quartic_vertices) = npt
    L = nbands(npt.swt)
    sys = swt.sys
    Q2 = zeros(ComplexF64, L, L)

    i = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            i += 1

            Q2_1 = quadratic_Q41_dipole(npt, bond, q, q_index, (0, 1, 0, 1); opts...)
            Q2_2 = quadratic_Q42_dipole(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
            Q2_3 = quadratic_Q41_dipole(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
            Q2_4 = quadratic_Q41_dipole(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
            Q2_5 = quadratic_Q43_dipole(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
            Q2_6 = quadratic_Q42_dipole(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
            Q2_7 = quadratic_Q41_dipole(npt, bond, q, q_index, (1, 0, 0, 0); opts...)
            Q2_8 = quadratic_Q41_dipole(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
            Q2_9 = quadratic_Q43_dipole(npt, bond, q, q_index, (1, 0, 0, 0); opts...)

            V41 = real_space_quartic_vertices[i].V41
            V42 = real_space_quartic_vertices[i].V42
            V43 = real_space_quartic_vertices[i].V43

            @. Q2 += V41 * Q2_1 + V42 * (Q2_2 + Q2_6) + V43 * (Q2_3 + Q2_7) + conj(V42) * (Q2_5 + Q2_9) + conj(V43) * (Q2_4 + Q2_8)

        end
    end

    return Q2
end

# # This function is an alternative function to quadratic_vertex_dipole, which should in principle solve the problem of divergence near the zero mode
# function quadratic_Q41_dipole_1(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
#     (; swt, Vps) = npt
#     L = nbands(swt)
#     Hk = zeros(ComplexF64, 2L, 2L)
#     Vk = zeros(ComplexF64, 2L, 2L)

#     αs = [bond.i, bond.j]
#     α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
#     Vq = view(Vps, :, :, q_index)
#     # 12 for the total number of terms to be integrated over
#     full_cartes_indices = CartesianIndices((1:L, 1:L, 1:12))
#     full_linear_indices = LinearIndices(full_cartes_indices)
#     num_ints = length(full_linear_indices)

#     # Here we perform an integration
#     ret = hcubature((0,0,0), (1,1,1); opts...) do k
#         f = zeros(ComplexF64, num_ints)
#         swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
#         bogoliubov!(Vk, Hk)
#         for n₁ in 1:L, n₂ in 1:L
#            for m in 1:L
#                i = full_linear_indices[n₁, n₂, 1]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₃+L, m]) * φ4([-q, k, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 2]
#                f[i] += Vk[α₃, m]   * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 3]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 4]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂, m])   * φ4([k, -k, -q, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 5]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃+L, m]) * φ4([k, -q, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 6]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂, m])   * φ4([k, -k, q, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 7]
#                f[i] += Vk[α₃, m]   * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 8]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 9]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₃+L, m]) * φ4([q, k, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 10]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃+L, m]) * φ4([k, q, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 11]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 12]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
#            end
#         end

#         return SVector{num_ints}(f)
#     end 

#     Q41_k = reshape(ret[1], (L, L, 12))
#     Q41 = zeros(ComplexF64, L, L)
#     for n₁ in 1:L, n₂ in 1:L
#         Q41[n₁, n₂] += conj(Vq[α₁, n₁]) * Vq[α₄, n₂] * Q41_k[n₁, n₂, 1] +
#         conj(Vq[α₁, n₁]) * Vq[α₂+L, n₂] * Q41_k[n₁, n₂, 2] +
#         conj(Vq[α₁, n₁]) * Vq[α₃, n₂] * Q41_k[n₁, n₂, 3] +
#         conj(Vq[α₃+L, n₁]) * Vq[α₄, n₂] * Q41_k[n₁, n₂, 4] +
#         conj(Vq[α₂, n₁]) * Vq[α₄, n₂] * Q41_k[n₁, n₂, 5] +
#         Vq[α₃, n₂] * conj(Vq[α₄+L, n₁]) * Q41_k[n₁, n₂, 6] +
#         Vq[α₁+L, n₂] * conj(Vq[α₂, n₁]) * Q41_k[n₁, n₂, 7] +
#         conj(Vq[α₂, n₁]) * Vq[α₃, n₂] * Q41_k[n₁, n₂, 8] +
#         Vq[α₁+L, n₂] * conj(Vq[α₄+L, n₁]) * Q41_k[n₁, n₂, 9] +
#         Vq[α₂+L, n₂] * conj(Vq[α₄+L, n₁]) * Q41_k[n₁, n₂, 10] +
#         Vq[α₁+L, n₂] * conj(Vq[α₃+L, n₁]) * Q41_k[n₁, n₂, 11] +
#         Vq[α₂+L, n₂] * conj(Vq[α₃+L, n₁]) * Q41_k[n₁, n₂, 12]
#     end

#     return Q41
# end


# function quadratic_Q42_dipole_1(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
#     (; swt, Vps) = npt
#     L = nbands(swt)
#     Hk = zeros(ComplexF64, 2L, 2L)
#     Vk = zeros(ComplexF64, 2L, 2L)

#     αs = [bond.i, bond.j]
#     α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
#     Vq = view(Vps, :, :, q_index)
#     # 12 for the total number of terms to be integrated over
#     full_cartes_indices = CartesianIndices((1:L, 1:L, 1:12))
#     full_linear_indices = LinearIndices(full_cartes_indices)
#     num_ints = length(full_linear_indices)

#     # Here we perform an integration
#     ret = hcubature((0,0,0), (1,1,1); opts...) do k
#         f = zeros(ComplexF64, num_ints)
#         swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
#         bogoliubov!(Vk, Hk)
#         for n₁ in 1:L, n₂ in 1:L
#            for m in 1:L
#                i = full_linear_indices[n₁, n₂, 1]
#                f[i] += Vk[α₂, m]   * conj(Vk[α₃+L, m]) * φ4([-q, k, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 2]
#                f[i] += Vk[α₃, m]   * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 3]
#                f[i] += Vk[α₂, m]   * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 4]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂+L, m]) * φ4([k, -k, -q, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 5]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃+L, m]) * φ4([k, -q, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 6]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂+L, m]) * φ4([k, -k, q, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 7]
#                f[i] += Vk[α₃, m]   * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 8]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 9]
#                f[i] += Vk[α₂, m]   * conj(Vk[α₃+L, m]) * φ4([q, k, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 10]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃+L, m]) * φ4([k, q, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 11]
#                f[i] += Vk[α₂, m]   * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 12]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
#            end
#         end

#         return SVector{num_ints}(f)
#     end 

#     Q42_k = reshape(ret[1], (L, L, 12))
#     Q42 = zeros(ComplexF64, L, L)
#     for n₁ in 1:L, n₂ in 1:L
#         Q42[n₁, n₂] += conj(Vq[α₁, n₁]) * Vq[α₄, n₂] * Q42_k[n₁, n₂, 1] +
#         conj(Vq[α₁, n₁]) * Vq[α₂, n₂] * Q42_k[n₁, n₂, 2] +
#         conj(Vq[α₁, n₁]) * Vq[α₃, n₂] * Q42_k[n₁, n₂, 3] +
#         conj(Vq[α₃+L, n₁]) * Vq[α₄, n₂] * Q42_k[n₁, n₂, 4] +
#         conj(Vq[α₂+L, n₁]) * Vq[α₄, n₂] * Q42_k[n₁, n₂, 5] +
#         Vq[α₃, n₂] * conj(Vq[α₄+L, n₁]) * Q42_k[n₁, n₂, 6] +
#         Vq[α₁+L, n₂] * conj(Vq[α₂+L, n₁]) * Q42_k[n₁, n₂, 7] +
#         conj(Vq[α₂+L, n₁]) * Vq[α₃, n₂] * Q42_k[n₁, n₂, 8] +
#         Vq[α₁+L, n₂] * conj(Vq[α₄+L, n₁]) * Q42_k[n₁, n₂, 9] +
#         Vq[α₂, n₂] * conj(Vq[α₄+L, n₁]) * Q42_k[n₁, n₂, 10] +
#         Vq[α₁+L, n₂] * conj(Vq[α₃+L, n₁]) * Q42_k[n₁, n₂, 11] +
#         Vq[α₂, n₂] * conj(Vq[α₃+L, n₁]) * Q42_k[n₁, n₂, 12]
#     end

#     return Q42
# end


# function quadratic_Q43_dipole_1(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
#     (; swt, Vps) = npt
#     L = nbands(swt)
#     Hk = zeros(ComplexF64, 2L, 2L)
#     Vk = zeros(ComplexF64, 2L, 2L)

#     αs = [bond.i, bond.j]
#     α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
#     Vq = view(Vps, :, :, q_index)
#     # 12 for the total number of terms to be integrated over
#     full_cartes_indices = CartesianIndices((1:L, 1:L, 1:12))
#     full_linear_indices = LinearIndices(full_cartes_indices)
#     num_ints = length(full_linear_indices)

#     # Here we perform an integration
#     ret = hcubature((0,0,0), (1,1,1); opts...) do k
#         f = zeros(ComplexF64, num_ints)
#         swt_hamiltonian_dipole!(Hk, swt, Vec3(k))
#         bogoliubov!(Vk, Hk)
#         for n₁ in 1:L, n₂ in 1:L
#            for m in 1:L
#                i = full_linear_indices[n₁, n₂, 1]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₃, m])   * φ4([-q, k, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 2]
#                f[i] += Vk[α₃+L, m] * conj(Vk[α₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 3]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 4]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂, m])   * φ4([k, -k, -q, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 5]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃, m])   * φ4([k, -q, -k, q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 6]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₂, m])   * φ4([k, -k, q, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 7]
#                f[i] += Vk[α₃+L, m] * conj(Vk[α₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 8]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 9]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₃, m])   * φ4([q, k, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 10]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₃, m])   * φ4([k, q, -k, -q], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 11]
#                f[i] += Vk[α₂+L, m] * conj(Vk[α₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n)
#                i = full_linear_indices[n₁, n₂, 12]
#                f[i] += Vk[α₁+L, m] * conj(Vk[α₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
#            end
#         end

#         return SVector{num_ints}(f)
#     end 

#     Q43_k = reshape(ret[1], (L, L, 12))
#     Q43 = zeros(ComplexF64, L, L)
#     for n₁ in 1:L, n₂ in 1:L
#         Q43[n₁, n₂] += conj(Vq[α₁, n₁]) * Vq[α₄, n₂] * Q43_k[n₁, n₂, 1] +
#         conj(Vq[α₁, n₁]) * Vq[α₂+L, n₂] * Q43_k[n₁, n₂, 2] +
#         conj(Vq[α₁, n₁]) * Vq[α₃+L, n₂] * Q43_k[n₁, n₂, 3] +
#         conj(Vq[α₃, n₁]) * Vq[α₄, n₂] * Q43_k[n₁, n₂, 4] +
#         conj(Vq[α₂, n₁]) * Vq[α₄, n₂] * Q43_k[n₁, n₂, 5] +
#         Vq[α₃+L, n₂] * conj(Vq[α₄+L, n₁]) * Q43_k[n₁, n₂, 6] +
#         Vq[α₁+L, n₂] * conj(Vq[α₂, n₁]) * Q43_k[n₁, n₂, 7] +
#         conj(Vq[α₂, n₁]) * Vq[α₃+L, n₂] * Q43_k[n₁, n₂, 8] +
#         Vq[α₁+L, n₂] * conj(Vq[α₄+L, n₁]) * Q43_k[n₁, n₂, 9] +
#         Vq[α₂+L, n₂] * conj(Vq[α₄+L, n₁]) * Q43_k[n₁, n₂, 10] +
#         Vq[α₁+L, n₂] * conj(Vq[α₃, n₁]) * Q43_k[n₁, n₂, 11] +
#         Vq[α₂+L, n₂] * conj(Vq[α₃, n₁]) * Q43_k[n₁, n₂, 12]
#     end

#     return Q43
# end


# function quadratic_vertex_dipole_1(npt::NonPerturbativeTheory, q::Vec3, q_index::CartesianIndex{3}; opts...)
#     (; swt, real_space_quartic_vertices) = npt
#     L = nbands(npt.swt)
#     sys = swt.sys
#     Q2 = zeros(ComplexF64, L, L)

#     i = 0
#     for int in sys.interactions_union
#         for coupling in int.pair
#             coupling.isculled && break
#             bond = coupling.bond
#             i += 1

#             Q2_1 = quadratic_Q41_dipole_1(npt, bond, q, q_index, (0, 1, 0, 1); opts...)
#             Q2_2 = quadratic_Q42_dipole_1(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
#             Q2_3 = quadratic_Q41_dipole_1(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
#             Q2_4 = quadratic_Q41_dipole_1(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
#             Q2_5 = quadratic_Q43_dipole_1(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
#             Q2_6 = quadratic_Q42_dipole_1(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
#             Q2_7 = quadratic_Q41_dipole_1(npt, bond, q, q_index, (1, 0, 0, 0); opts...)
#             Q2_8 = quadratic_Q41_dipole_1(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
#             Q2_9 = quadratic_Q43_dipole_1(npt, bond, q, q_index, (1, 0, 0, 0); opts...)

#             V41 = real_space_quartic_vertices[i].V41
#             V42 = real_space_quartic_vertices[i].V42
#             V43 = real_space_quartic_vertices[i].V43

#             @. Q2 += V41 * Q2_1 + V42 * (Q2_2 + Q2_6) + V43 * (Q2_3 + Q2_7) + conj(V42) * (Q2_5 + Q2_9) + conj(V43) * (Q2_4 + Q2_8)

#         end
#     end

#     return Q2
# end