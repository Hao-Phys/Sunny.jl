
function quartic_U40_dipole!(U41_buf::Array{ComplexF64, 4}, npt::NonPerturbativeTheory,qs_indices::Vector{CartesianIndex{3}}, α::Int)
    U41_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])
    Vp4 = view(Vps, :, :, qs_indices[4])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U41_buf[n₁, n₂, n₃, n₄] += conj(Vp1[α, n₁]) * conj(Vp2[α, n₂]) * Vp3[α, n₃] * Vp4[α, n₄]  +
        conj(Vp1[α, n₁]) * Vp3[α+L, n₃] * conj(Vp2[α+L, n₂]) * Vp4[α, n₄] +
        conj(Vp1[α, n₁]) * Vp4[α+L, n₄] * Vp3[α, n₃] * conj(Vp2[α+L, n₂]) +
        Vp3[α+L, n₃] * conj(Vp2[α, n₂]) * conj(Vp1[α+L, n₁]) * Vp4[α, n₄] +
        Vp4[α+L, n₄] * conj(Vp2[α, n₂]) * Vp3[α, n₃] * conj(Vp1[α+L, n₁]) +
        Vp3[α+L, n₃] * Vp4[α+L, n₄] * conj(Vp1[α+L, n₁]) * conj(Vp2[α+L, n₂])
    end
end

function quartic_U40_symmetrized_dipole(npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    swt = npt.swt
    L = nbands(swt)
    iq₁, iq₂, iq₃, iq₄ = qs_indices

    U4 = zeros(ComplexF64, L, L, L, L)
    U4_buf = zeros(ComplexF64, L, L, L, L)
    # A new buffer to hold the permuted results. According to Julia docs "No in-place permutation is supported and unexpected results will happen if src and dest have overlapping memory regions."
    U4_buf_perm = zeros(ComplexF64, L, L, L, L)

    quartic_U40_dipole!(U4_buf, npt, qs_indices, α)
    U4 .+= U4_buf

    quartic_U40_dipole!(U4_buf, npt,[iq₂, iq₁, iq₃, iq₄], α)
    permutedims!(U4_buf_perm, U4_buf, (2, 1, 3, 4))
    U4 .+= U4_buf_perm

    quartic_U40_dipole!(U4_buf, npt,[iq₂, iq₁, iq₃, iq₄], α)
    permutedims!(U4_buf_perm, U4_buf, (1, 2, 4, 3))
    U4 .+= U4_buf_perm

    quartic_U40_dipole!(U4_buf, npt,[iq₂, iq₁, iq₃, iq₄], α)
    permutedims!(U4_buf_perm, U4_buf, (2, 1, 4, 3))
    U4 .+= U4_buf_perm
end

function quartic_U41_dipole!(U41_buf::Array{ComplexF64, 4}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U41_buf .= 0.0
    (; swt, Vps) = npt
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

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U41_buf[n₁, n₂, n₃, n₄] += conj(Vp1[α₁, n₁]) * conj(Vp2[α₂, n₂]) * Vp3[α₃, n₃] * Vp4[α₄, n₄] * phase1 +
        conj(Vp1[α₁, n₁]) * Vp3[α₂+L, n₃] * conj(Vp2[α₃+L, n₂]) * Vp4[α₄, n₄] * phase2 +
        conj(Vp1[α₁, n₁]) * Vp4[α₂+L, n₄] * Vp3[α₃, n₃] * conj(Vp2[α₄+L, n₂]) * phase3 +
        Vp3[α₁+L, n₃] * conj(Vp2[α₂, n₂]) * conj(Vp1[α₃+L, n₁]) * Vp4[α₄, n₄] * phase4 +
        Vp4[α₁+L, n₄] * conj(Vp2[α₂, n₂]) * Vp3[α₃, n₃] * conj(Vp1[α₄+L, n₁]) * phase5 +
        Vp3[α₁+L, n₃] * Vp4[α₂+L, n₄] * conj(Vp1[α₃+L, n₁]) * conj(Vp2[α₄+L, n₂]) * phase6 
    end
end

function quartic_U42_dipole!(U42_buf::Array{ComplexF64, 4}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U42_buf .= 0.0
    (; swt, Vps) = npt
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

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U42_buf[n₁, n₂, n₃, n₄] += conj(Vp1[α₁, n₁]) * conj(Vp2[α₂+L, n₂]) * Vp3[α₃, n₃] * Vp4[α₄, n₄] * phase1 +
        conj(Vp1[α₁, n₁]) * Vp3[α₂, n₃] * conj(Vp2[α₃+L, n₂]) * Vp4[α₄, n₄] * phase2 +
        conj(Vp1[α₁, n₁]) * Vp4[α₂, n₄] * Vp3[α₃, n₃] * conj(Vp2[α₄+L, n₂]) * phase3 +
        Vp3[α₁+L, n₃] * conj(Vp2[α₂+L, n₂]) * conj(Vp1[α₃+L, n₁]) * Vp4[α₄, n₄] * phase4 +
        Vp4[α₁+L, n₄] * conj(Vp2[α₂+L, n₂]) * Vp3[α₃, n₃] * conj(Vp1[α₄+L, n₁]) * phase5 +
        Vp3[α₁+L, n₃] * Vp4[α₂, n₄] * conj(Vp1[α₃+L, n₁]) * conj(Vp2[α₄+L, n₂]) * phase6 
    end
end

function quartic_U43_dipole!(U43_buf::Array{ComplexF64, 4}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    U43_buf .= 0.0
    (; swt, Vps) = npt
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

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L, n₄ in 1:L
        U43_buf[n₁, n₂, n₃, n₄] += conj(Vp1[α₁, n₁]) * conj(Vp2[α₂, n₂]) * Vp3[α₃+L, n₃] * Vp4[α₄, n₄] * phase1 +
        conj(Vp1[α₁, n₁]) * Vp3[α₂+L, n₃] * conj(Vp2[α₃, n₂]) * Vp4[α₄, n₄] * phase2 +
        conj(Vp1[α₁, n₁]) * Vp4[α₂+L, n₄] * Vp3[α₃+L, n₃] * conj(Vp2[α₄+L, n₂]) * phase3 +
        Vp3[α₁+L, n₃] * conj(Vp2[α₂, n₂]) * conj(Vp1[α₃, n₁]) * Vp4[α₄, n₄] * phase4 +
        Vp4[α₁+L, n₄] * conj(Vp2[α₂, n₂]) * Vp3[α₃+L, n₃] * conj(Vp1[α₄+L, n₁]) * phase5 +
        Vp3[α₁+L, n₃] * Vp4[α₂+L, n₄] * conj(Vp1[α₃, n₁]) * conj(Vp2[α₄+L, n₂]) * phase6 
    end
end

function quartic_U4_symmetrized_dipole(quartic_fun::Function, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{4, Int})
    swt = npt.swt
    L = nbands(swt)
    q₁, q₂, q₃, q₄ = view(qs, :)
    iq₁, iq₂, iq₃, iq₄ = qs_indices

    U4 = zeros(ComplexF64, L, L, L, L)
    U4_buf = zeros(ComplexF64, L, L, L, L)
    # A new buffer to hold the permuted results. According to Julia docs "No in-place permutation is supported and unexpected results will happen if src and dest have overlapping memory regions."
    U4_buf_perm = zeros(ComplexF64, L, L, L, L)

    quartic_fun(U4_buf, npt, bond, qs, qs_indices, φas)
    U4 .+= U4_buf

    quartic_fun(U4_buf, npt, bond, [q₂, q₁, q₃, q₄], [iq₂, iq₁, iq₃, iq₄], φas)
    permutedims!(U4_buf_perm, U4_buf, (2, 1, 3, 4))
    U4 .+= U4_buf_perm

    quartic_fun(U4_buf, npt, bond, [q₁, q₂, q₄, q₃], [iq₁, iq₂, iq₄, iq₃], φas)
    permutedims!(U4_buf_perm, U4_buf, (1, 2, 4, 3))
    U4 .+= U4_buf_perm

    quartic_fun(U4_buf, npt, bond, [q₂, q₁, q₄, q₃], [iq₂, iq₁, iq₄, iq₃], φas)
    permutedims!(U4_buf_perm, U4_buf, (2, 1, 4, 3))
    U4 .+= U4_buf_perm

    return U4
end

function quartic_vertex_dipole(npt::NonPerturbativeTheory, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}})
    (; swt, real_space_quartic_vertices) = npt
    (; sys, data) = swt
    (; stevens_coefs) = data

    L = nbands(swt)

    for i in 1:L
        (; c4, c6) = stevens_coefs[i]
        @assert iszero(c4) "Higher rank Stevens operators not supported in :dipole_large_S non-perturbative calculations"
        @assert iszero(c6) "Higher rank Stevens operators not supported in :dipole_large_S non-perturbative calculations"
    end

    U4 = zeros(ComplexF64, L, L, L, L)

    # For dipole mode, we first loop over all interactions
    i = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            i += 1

            U4_1 = quartic_U4_symmetrized_dipole(quartic_U41_dipole!, npt, bond, qs, qs_indices, (0, 1, 0, 1))
            U4_2 = quartic_U4_symmetrized_dipole(quartic_U42_dipole!, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            U4_3 = quartic_U4_symmetrized_dipole(quartic_U41_dipole!, npt, bond, qs, qs_indices, (1, 1, 1, 0))
            U4_4 = quartic_U4_symmetrized_dipole(quartic_U41_dipole!, npt, bond, qs, qs_indices, (0, 1, 1, 1))
            U4_5 = quartic_U4_symmetrized_dipole(quartic_U43_dipole!, npt, bond, qs, qs_indices, (0, 1, 1, 1))
            U4_6 = quartic_U4_symmetrized_dipole(quartic_U42_dipole!, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            U4_7 = quartic_U4_symmetrized_dipole(quartic_U41_dipole!, npt, bond, qs, qs_indices, (1, 0, 0, 0))
            U4_8 = quartic_U4_symmetrized_dipole(quartic_U41_dipole!, npt, bond, qs, qs_indices, (0, 0, 0, 1))
            U4_9 = quartic_U4_symmetrized_dipole(quartic_U43_dipole!, npt, bond, qs, qs_indices, (1, 0, 0, 0))

            V41 = real_space_quartic_vertices[i].V41
            V42 = real_space_quartic_vertices[i].V42
            V43 = real_space_quartic_vertices[i].V43

            @. U4 += V41 * U4_1 + V42 * (U4_2 + U4_6) + V43 * (U4_3 + U4_7) + conj(V42) * (U4_5 + U4_9) + conj(V43) * (U4_4 + U4_8)

        end
    end

    # For dipole mode, we should also loop over Stevens operators of rank two. In particular, 𝒪[2,0]
    S = (sys.Ns[1]-1)/2
    for i in 1:L
        c2 = stevens_coefs[i].c2[3]
        if !iszero(c2)
            # Use the unrenormalize vertex function for the :dipole mode
            U4_0 = quartic_U40_symmetrized_dipole(npt, qs_indices, i)
            factor 1/(1-1/(2S))
            @. U4 += 3c2 * U4_0 * factor
        end
    end

    return U4
end