# Given the `npt`, and a `pc`, and a series of `qs` and their indices `qs_indices`, and `φas` return to the quadratic vertex from normal-ordering of the quartic vertex. The results return to
# (N-1) × (N-1) × (N-1) × (N-1) × L × L array
function quadratic_Q41_SUN(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:nflavors, 1:nflavors, 1:nflavors, 1:nflavors, 1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_SUN!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           σ₁, σ₂, σ₃, σ₄, n₁, n₂ = cart[1], cart[2], cart[3], cart[4], cart[5], cart[6]
           for m in 1:L
               f[i] += conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂+L, m] * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q41_buf = reshape(ret[1], (nflavors, nflavors, nflavors, nflavors, L, L))

    return Q41_buf
end

function quadratic_Q42_SUN(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:nflavors, 1:nflavors, 1:nflavors, 1:nflavors, 1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_SUN!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           σ₁, σ₂, σ₃, σ₄, n₁, n₂ = cart[1], cart[2], cart[3], cart[4], cart[5], cart[6]
           for m in 1:L
               f[i] += conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vq[(α₂-1)*nflavors+σ₂, n₂] * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂, m] * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂+L, m]) * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂+L, n₁]) * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂+L, m]) * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * conj(Vq[(α₂-1)*nflavors+σ₂+L, n₁]) * Vk[(α₃-1)*nflavors+σ₃, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂+L, n₁]) * Vq[(α₃-1)*nflavors+σ₃, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂, m] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂, n₂] * conj(Vk[(α₃-1)*nflavors+σ₃+L, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂, m] * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂, n₂] * conj(Vq[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q42_buf = reshape(ret[1], (nflavors, nflavors, nflavors, nflavors, L, L))

    return Q42_buf
end


function quadratic_Q43_SUN(npt::NonPerturbativeTheory, bond::Bond, q::Vec3, q_index::CartesianIndex{3}, φas::NTuple{4, Int}; opts...)
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    Hk = zeros(ComplexF64, 2L, 2L)
    Vk = zeros(ComplexF64, 2L, 2L)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]
    Vq = view(Vps, :, :, q_index)
    cartes_indices = CartesianIndices((1:nflavors, 1:nflavors, 1:nflavors, 1:nflavors, 1:L, 1:L))
    linear_indices = LinearIndices(cartes_indices)
    num_ints = length(cartes_indices)

    # Here we perform an integration
    ret = hcubature((0,0,0), (1,1,1); opts...) do k
        f = zeros(ComplexF64, num_ints)
        swt_hamiltonian_SUN!(Hk, swt, Vec3(k))
        bogoliubov!(Vk, Hk)
        for cart in cartes_indices
           i = linear_indices[cart] 
           σ₁, σ₂, σ₃, σ₄, n₁, n₂ = cart[1], cart[2], cart[3], cart[4], cart[5], cart[6]
           for m in 1:L
               f[i] += conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([-q, k, -k, q], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * Vk[(α₃-1)*nflavors+σ₃+L, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, q, k, -k], φas, bond.n) +
               conj(Vq[(α₁-1)*nflavors+σ₁, n₁]) * Vk[(α₂-1)*nflavors+σ₂+L, m] * Vq[(α₃-1)*nflavors+σ₃+L, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([-q, k, q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * conj(Vq[(α₃-1)*nflavors+σ₃, n₁]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -k, -q, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * Vq[(α₄-1)*nflavors+σ₄, n₂] * φ4([k, -q, -k, q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vk[(α₂-1)*nflavors+σ₂, m]) * Vq[(α₃-1)*nflavors+σ₃+L, n₂] * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, -k, q, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * Vk[(α₃-1)*nflavors+σ₃+L, m] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, -q, k, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * conj(Vq[(α₂-1)*nflavors+σ₂, n₁]) * Vq[(α₃-1)*nflavors+σ₃+L, n₂] * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, -q, q, -k], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([q, k, -k, -q], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * conj(Vk[(α₃-1)*nflavors+σ₃, m]) * conj(Vq[(α₄-1)*nflavors+σ₄+L, n₁]) * φ4([k, q, -k, -q], φas, bond.n) +
               Vq[(α₁-1)*nflavors+σ₁+L, n₂] * Vk[(α₂-1)*nflavors+σ₂+L, m] * conj(Vq[(α₃-1)*nflavors+σ₃, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([q, k, -q, -k], φas, bond.n) +
               Vk[(α₁-1)*nflavors+σ₁+L, m] * Vq[(α₂-1)*nflavors+σ₂+L, n₂] * conj(Vq[(α₃-1)*nflavors+σ₃, n₁]) * conj(Vk[(α₄-1)*nflavors+σ₄+L, m]) * φ4([k, q, -q, -k], φas, bond.n)
           end
        end

        return SVector{num_ints}(f)
    end 

    Q43_buf = reshape(ret[1], (nflavors, nflavors, nflavors, nflavors, L, L))

    return Q43_buf
end

function quadratic_vertex_SUN(npt::NonPerturbativeTheory, q::Vec3, q_index::CartesianIndex{3}; opts...)
    (; swt, real_space_quartic_vertices) = npt
    L = nbands(npt.swt)
    sys = swt.sys
    Q2 = zeros(ComplexF64, L, L)
    Q2_buf = zeros(ComplexF64, L, L)
    i = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            Q2_buf .= 0.0
            i += 1

            Q2_1 = quadratic_Q41_SUN(npt, bond, q, q_index, (0, 1, 0, 1); opts...)
            Q2_2 = quadratic_Q42_SUN(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
            Q2_3 = quadratic_Q42_SUN(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
            Q2_4 = quadratic_Q43_SUN(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
            Q2_5 = quadratic_Q43_SUN(npt, bond, q, q_index, (1, 0, 0, 0); opts...)
            Q2_6 = quadratic_Q41_SUN(npt, bond, q, q_index, (0, 1, 1, 1); opts...)
            Q2_7 = quadratic_Q41_SUN(npt, bond, q, q_index, (0, 0, 0, 1); opts...)
            Q2_8 = quadratic_Q41_SUN(npt, bond, q, q_index, (1, 1, 1, 0); opts...)
            Q2_9 = quadratic_Q41_SUN(npt, bond, q, q_index, (1, 0, 0, 0); opts...)

            V41 = real_space_quartic_vertices[i].V41
            V42 = real_space_quartic_vertices[i].V42
            V43 = real_space_quartic_vertices[i].V43

            @tensor begin
                Q2_buf[n₁, n₂] = V41[σ₁, σ₂, σ₃, σ₄] * Q2_1[σ₁, σ₃, σ₂, σ₄, n₁, n₂] +
                V42[σ₁, σ₃] * (Q2_2[σ₂, σ₂, σ₃, σ₁, n₁, n₂] + Q2_3[σ₂, σ₂, σ₁, σ₃, n₁, n₂]) +
                conj(V42[σ₁, σ₃]) * (Q2_4[σ₁, σ₃, σ₂, σ₂, n₁, n₂] + Q2_5[σ₃, σ₁, σ₂, σ₂, n₁, n₂]) +
                V43[σ₁, σ₃] * (Q2_6[σ₁, σ₂, σ₂, σ₃, n₁, n₂] + Q2_7[σ₁, σ₂, σ₂, σ₃, n₁, n₂]) +
                conj(V43[σ₁, σ₃]) * (Q2_8[σ₃, σ₂, σ₂, σ₁, n₁, n₂] + Q2_9[σ₃, σ₂, σ₂, σ₁, n₁, n₂])
            end

            Q2 .+= Q2_buf
        end
    end

    return Q2
end