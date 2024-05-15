function φ4(qs::Vector{Vec3}, φas::NTuple{4, Int}, n)
    ret = 1.0 + 0.0im
    for i in 1:4
        ret *= exp(2π*im * φas[i] * dot(qs[i], n))
    end
    return ret
end


# Given the `npt`, and a `pc`, and a series of `qs` and their indices `qs_indices`, and `φas` return to the quartic vertex
# (N-1) × (N-1) × (N-1) × (N-1) × L × L × L × L array
function quartic_U41_SUN!(U41_buf::Array{ComplexF64, 8}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U41_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    q₁, q₂, q₃, q₄ = view(qs, :)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    phase1 = φ4([q₁, q₂, q₃, q₄], φas, bond.n)
    phase2 = φ4([q₁, q₃, q₂, q₄], φas, bond.n)
    phase3 = φ4([q₁, q₄, q₃, q₂], φas, bond.n)
    phase4 = φ4([q₃, q₂, q₁, q₄], φas, bond.n)
    phase5 = φ4([q₄, q₂, q₃, q₁], φas, bond.n)
    phase6 = φ4([q₃, q₄, q₁, q₂], φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])
    Vp4 = view(Vps, :, :, qs_indices[4])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U41_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂, n₃, n₄] += conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase1 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp3[(α₂-1)*nflavors+σ₂+L, n₃] * conj(Vp2[(α₃-1)*nflavors+σ₃+L, n₂]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase2 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp4[(α₂-1)*nflavors+σ₂+L, n₄] * Vp3[(α₃-1)*nflavors+σ₃, n₃] * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase3 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase4 +
        Vp4[(α₁-1)*nflavors+σ₁+L, n₄] * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * conj(Vp1[(α₄-1)*nflavors+σ₄+L, n₁]) * phase5 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * Vp4[(α₂-1)*nflavors+σ₂+L, n₄] * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase6 
    end
end

function quartic_U42_SUN!(U42_buf::Array{ComplexF64, 8}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U42_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    q₁, q₂, q₃, q₄ = view(qs, :)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    phase1 = φ4([q₁, q₂, q₃, q₄], φas, bond.n)
    phase2 = φ4([q₁, q₃, q₂, q₄], φas, bond.n)
    phase3 = φ4([q₁, q₄, q₃, q₂], φas, bond.n)
    phase4 = φ4([q₃, q₂, q₁, q₄], φas, bond.n)
    phase5 = φ4([q₄, q₂, q₃, q₁], φas, bond.n)
    phase6 = φ4([q₃, q₄, q₁, q₂], φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])
    Vp4 = view(Vps, :, :, qs_indices[4])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U42_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂, n₃, n₄] += conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * conj(Vp2[(α₂-1)*nflavors+σ₂+L, n₂]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase1 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp3[(α₂-1)*nflavors+σ₂, n₃] * conj(Vp2[(α₃-1)*nflavors+σ₃+L, n₂]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase2 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp4[(α₂-1)*nflavors+σ₂, n₄] * Vp3[(α₃-1)*nflavors+σ₃, n₃] * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase3 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * conj(Vp2[(α₂-1)*nflavors+σ₂+L, n₂]) * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase4 +
        Vp4[(α₁-1)*nflavors+σ₁+L, n₄] * conj(Vp2[(α₂-1)*nflavors+σ₂+L, n₂]) * Vp3[(α₃-1)*nflavors+σ₃, n₃] * conj(Vp1[(α₄-1)*nflavors+σ₄+L, n₁]) * phase5 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * Vp4[(α₂-1)*nflavors+σ₂, n₄] * conj(Vp1[(α₃-1)*nflavors+σ₃+L, n₁]) * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase6 
    end
end

function quartic_U43_SUN!(U43_buf::Array{ComplexF64, 8}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U43_buf .= 0.0
    (; swt, Vps) = npt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(swt)
    q₁, q₂, q₃, q₄ = view(qs, :)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃, α₄ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1], αs[φas[4]+1]

    phase1 = φ4([q₁, q₂, q₃, q₄], φas, bond.n)
    phase2 = φ4([q₁, q₃, q₂, q₄], φas, bond.n)
    phase3 = φ4([q₁, q₄, q₃, q₂], φas, bond.n)
    phase4 = φ4([q₃, q₂, q₁, q₄], φas, bond.n)
    phase5 = φ4([q₄, q₂, q₃, q₁], φas, bond.n)
    phase6 = φ4([q₃, q₄, q₁, q₂], φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])
    Vp4 = view(Vps, :, :, qs_indices[4])

    for σ₁ in 1:nflavors, σ₂ in 1:nflavors, σ₃ in 1:nflavors, σ₄ in 1:nflavors, n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U43_buf[σ₁, σ₂, σ₃, σ₄, n₁, n₂, n₃, n₄] += conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * Vp3[(α₃-1)*nflavors+σ₃+L, n₃] * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase1 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp3[(α₂-1)*nflavors+σ₂+L, n₃] * conj(Vp2[(α₃-1)*nflavors+σ₃, n₂]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase2 +
        conj(Vp1[(α₁-1)*nflavors+σ₁, n₁]) * Vp4[(α₂-1)*nflavors+σ₂+L, n₄] * Vp3[(α₃-1)*nflavors+σ₃+L, n₃] * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase3 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * conj(Vp1[(α₃-1)*nflavors+σ₃, n₁]) * Vp4[(α₄-1)*nflavors+σ₄, n₄] * phase4 +
        Vp4[(α₁-1)*nflavors+σ₁+L, n₄] * conj(Vp2[(α₂-1)*nflavors+σ₂, n₂]) * Vp3[(α₃-1)*nflavors+σ₃+L, n₃] * conj(Vp1[(α₄-1)*nflavors+σ₄+L, n₁]) * phase5 +
        Vp3[(α₁-1)*nflavors+σ₁+L, n₃] * Vp4[(α₂-1)*nflavors+σ₂+L, n₄] * conj(Vp1[(α₃-1)*nflavors+σ₃, n₁]) * conj(Vp2[(α₄-1)*nflavors+σ₄+L, n₂]) * phase6 
    end
end

function quartic_U4_symmetrized_SUN(quartic_fun::Function, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    swt = npt.swt
    N = swt.sys.Ns[1]
    nflavors = N - 1
    L = nbands(npt.swt)
    q₁, q₂, q₃, q₄ = view(qs, :)
    iq₁, iq₂, iq₃, iq₄ = qs_indices

    U4 = zeros(ComplexF64, nflavors, nflavors, nflavors, nflavors, L, L, L, L)
    U4_buf = zeros(ComplexF64, nflavors, nflavors, nflavors, nflavors, L, L, L, L)
    # A new buffer to hold the permuted results. According to Julia docs "No in-place permutation is supported and unexpected results will happen if src and dest have overlapping memory regions."
    U4_buf_perm = zeros(ComplexF64, nflavors, nflavors, nflavors, nflavors, L, L, L, L)

    quartic_fun(U4_buf, npt, bond, qs, qs_indices, φas)
    U4 .+= U4_buf

    quartic_fun(U4_buf, npt, bond, [q₂, q₁, q₃, q₄], [iq₂, iq₁, iq₃, iq₄], φas)
    permutedims!(U4_buf_perm, U4_buf, (1, 2, 3, 4, 6, 5, 7, 8))
    U4 .+= U4_buf_perm

    quartic_fun(U4_buf, npt, bond, [q₁, q₂, q₄, q₃], [iq₁, iq₂, iq₄, iq₃], φas)
    permutedims!(U4_buf_perm, U4_buf, (1, 2, 3, 4, 5, 6, 8, 7))
    U4 .+= U4_buf_perm

    quartic_fun(U4_buf, npt, bond, [q₂, q₁, q₄, q₃], [iq₂, iq₁, iq₄, iq₃], φas)
    permutedims!(U4_buf_perm, U4_buf, (1, 2, 3, 4, 6, 5, 8, 7))
    U4 .+= U4_buf_perm

    return U4
end

function quartic_vertex_SUN(npt::NonPerturbativeTheory, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}})
    (; swt, real_space_quartic_vertices, clustersize) = npt
    L = nbands(npt.swt)
    sys = swt.sys

    U4 = zeros(ComplexF64, L, L, L, L)
    U4_buf = zeros(ComplexF64, L, L, L, L)

    i = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            U4_buf .= 0.0
            i += 1

            U4_1 = quartic_U4_symmetrized_SUN(quartic_U41_SUN!, npt, bond, qs, qs_indices, (0, 1, 0, 1))
            U4_2 = quartic_U4_symmetrized_SUN(quartic_U42_SUN!, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            U4_3 = quartic_U4_symmetrized_SUN(quartic_U42_SUN!, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            U4_4 = quartic_U4_symmetrized_SUN(quartic_U43_SUN!, npt, bond, qs, qs_indices, (1, 1, 0, 1))
            U4_5 = quartic_U4_symmetrized_SUN(quartic_U43_SUN!, npt, bond, qs, qs_indices, (1, 0, 0, 0))
            U4_6 = quartic_U4_symmetrized_SUN(quartic_U41_SUN!, npt, bond, qs, qs_indices, (0, 1, 1, 1))
            U4_7 = quartic_U4_symmetrized_SUN(quartic_U41_SUN!, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            U4_8 = quartic_U4_symmetrized_SUN(quartic_U41_SUN!, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            U4_9 = quartic_U4_symmetrized_SUN(quartic_U41_SUN!, npt, bond, qs, qs_indices, (1, 0, 0, 0))

            V41 = real_space_quartic_vertices[i].V41
            V42 = real_space_quartic_vertices[i].V42
            V43 = real_space_quartic_vertices[i].V43

            @tensor begin
                U4_buf[n₁, n₂, n₃, n₄] = V41[σ₁, σ₂, σ₃, σ₄] * U4_1[σ₁, σ₃, σ₂, σ₄, n₁, n₂, n₃, n₄] +
                V42[σ₁, σ₃] * (U4_2[σ₂, σ₂, σ₃, σ₁, n₁, n₂, n₃, n₄] + U4_3[σ₂, σ₂, σ₁, σ₃, n₁, n₂, n₃, n₄]) +
                conj(V42[σ₁, σ₃]) * (U4_4[σ₃, σ₂, σ₁, σ₂, n₁, n₂, n₃, n₄] + U4_5[σ₃, σ₁, σ₂, σ₂, n₁, n₂, n₃, n₄]) +
                V43[σ₁, σ₃] * (U4_6[σ₁, σ₂, σ₂, σ₃, n₁, n₂, n₃, n₄] + U4_7[σ₁, σ₂, σ₂, σ₃, n₁, n₂, n₃, n₄]) +
                conj(V43[σ₁, σ₃]) * (U4_8[σ₃, σ₂, σ₂, σ₁, n₁, n₂, n₃, n₄] + U4_9[σ₃, σ₂, σ₂, σ₁, n₁, n₂, n₃, n₄])
            end

            U4 .+= U4_buf
        end
    end

    return U4 / (clustersize[1]*clustersize[2]*clustersize[3])
end