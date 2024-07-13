function cubic_U31_dipole!(U31_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{3, Int})
    U31_buf .= 0.0
    (; swt, Vps) = npt

    L = nbands(swt)
    q₁, q₂, q₃ = view(qs, :)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1]

    phase1 = φ3([q₁, q₂, q₃], φas, bond.n)
    phase2 = φ3([q₂, q₁, q₃], φas, bond.n)
    phase3 = φ3([q₃, q₂, q₁], φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U31_buf[n₁, n₂, n₃] += conj(Vp1[α₁, n₁]) * Vp2[α₂, n₂] * Vp3[α₃, n₃] * phase1 +
        Vp2[α₁+L, n₂] * conj(Vp1[α₂+L, n₁]) * Vp3[α₃, n₃] * phase2 +
        Vp3[α₁+L, n₃] * Vp2[α₂, n₂] * conj(Vp1[α₃+L, n₁]) * phase3
    end
end

function cubic_U32_dipole!(U32_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{3, Int})
    U32_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)
    q₁, q₂, q₃ = view(qs, :)

    αs = [bond.i, bond.j]
    α₁, α₂, α₃ = αs[φas[1]+1], αs[φas[2]+1], αs[φas[3]+1]

    phase1 = φ3([q₁, q₂, q₃], φas, bond.n)
    phase2 = φ3([q₂, q₁, q₃], φas, bond.n)
    phase3 = φ3([q₃, q₂, q₁], φas, bond.n)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U32_buf[n₁, n₂, n₃] += conj(Vp1[α₁, n₁]) * Vp2[α₂+L, n₂] * Vp3[α₃, n₃] * phase1 +
        Vp2[α₁+L, n₂] * conj(Vp1[α₂, n₁]) * Vp3[α₃, n₃] * phase2 +
        Vp3[α₁+L, n₃] * Vp2[α₂+L, n₂] * conj(Vp1[α₃+L, n₁]) * phase3
    end
end

function cubic_U3_symmetrized_dipole(cubic_fun::Function, npt::NonPerturbativeTheory, bond::Bond, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}}, φas::NTuple{3, Int})
    swt = npt.swt
    L = nbands(swt)
    q₁, q₂, q₃ = view(qs, :)
    iq₁, iq₂, iq₃ = qs_indices

    U3 = zeros(ComplexF64, L, L, L)
    U3_buf = zeros(ComplexF64, L, L, L)
    # A new buffer to hold the permuted results. According to Julia docs "No in-place permutation is supported and unexpected results will happen if src and dest have overlapping memory regions."
    U3_buf_perm = zeros(ComplexF64, L, L, L)

    cubic_fun(U3_buf, npt, bond, qs, qs_indices, φas)
    U3 .+= U3_buf

    cubic_fun(U3_buf, npt, bond, [q₁, q₃, q₂], [iq₁, iq₃, iq₂], φas)
    permutedims!(U3_buf_perm, U3_buf, (1, 3, 2))
    U3 .+= U3_buf_perm

    return U3
end

function cubic_U31′_dipole!(U31_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    U31_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U31_buf[n₁, n₂, n₃] += conj(Vp1[α, n₁]) * Vp2[α, n₂] * Vp3[α, n₃] +
        Vp2[α+L, n₂] * conj(Vp1[α+L, n₁]) * Vp3[α, n₃] +
        Vp3[α+L, n₃] * Vp2[α, n₂] * conj(Vp1[α+L, n₁])
    end
end

function cubic_U32′_dipole!(U32_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    U32_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U32_buf[n₁, n₂, n₃] += conj(Vp1[α, n₁]) * Vp2[α+L, n₂] * Vp3[α, n₃] +
        Vp2[α+L, n₂] * conj(Vp1[α, n₁]) * Vp3[α, n₃] +
        Vp3[α+L, n₃] * Vp2[α+L, n₂] * conj(Vp1[α+L, n₁])
    end
end

function cubic_U33′_dipole!(U33_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    U33_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U33_buf[n₁, n₂, n₃] += conj(Vp1[α+L, n₁]) * Vp2[α, n₂] * Vp3[α, n₃] +
        Vp2[α, n₂] * conj(Vp1[α+L, n₁]) * Vp3[α, n₃] +
        Vp3[α, n₃] * Vp2[α, n₂] * conj(Vp1[α+L, n₁])
    end
end


function cubic_U34′_dipole!(U34_buf::Array{ComplexF64, 3}, npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    U34_buf .= 0.0
    (; swt, Vps) = npt
    L = nbands(swt)

    Vp1 = view(Vps, :, :, qs_indices[1])
    Vp2 = view(Vps, :, :, qs_indices[2])
    Vp3 = view(Vps, :, :, qs_indices[3])

    for n₁ in 1:L, n₂ in 1:L, n₃ in 1:L
        U34_buf[n₁, n₂, n₃] += conj(Vp1[α, n₁]) * Vp2[α+L, n₂] * Vp3[α+L, n₃] +
        Vp2[α+L, n₂] * conj(Vp1[α, n₁]) * Vp3[α+L, n₃] +
        Vp3[α+L, n₃] * Vp2[α+L, n₂] * conj(Vp1[α, n₁])
    end
end

function cubic_U3′_symmetrized_dipole(cubic_fun::Function, npt::NonPerturbativeTheory, qs_indices::Vector{CartesianIndex{3}}, α::Int)
    swt = npt.swt
    L = nbands(swt)
    iq₁, iq₂, iq₃ = qs_indices

    U3 = zeros(ComplexF64, L, L, L)
    U3_buf = zeros(ComplexF64, L, L, L)
    # A new buffer to hold the permuted results. According to Julia docs "No in-place permutation is supported and unexpected results will happen if src and dest have overlapping memory regions."
    U3_buf_perm = zeros(ComplexF64, L, L, L)

    cubic_fun(U3_buf, npt, qs_indices, α)
    U3 .+= U3_buf

    cubic_fun(U3_buf, npt, [iq₁, iq₃, iq₂], α)
    permutedims!(U3_buf_perm, U3_buf, (1, 3, 2))
    U3 .+= U3_buf_perm

    return U3
end

function cubic_vertex_dipole(npt::NonPerturbativeTheory, qs::Vector{Vec3}, qs_indices::Vector{CartesianIndex{3}})
    (; swt, real_space_cubic_vertices) = npt
    (; sys, data) = swt
    (; stevens_coefs) = data 

    L = nbands(npt.swt)

    for i in 1:L
        (; c6) = stevens_coefs[i]
        @assert iszero(c6) "Rank 6 Stevens operators not supported in :dipole non-perturbative calculations yet"
    end

    U3 = zeros(ComplexF64, L, L, L)

    i = 0
    # For this moment, we only support the cubic vertices from bilinear interactions for the dipole mode
    # We will implement the onsite terms later
    for int in sys.interactions_union 
        for coupling in int.pair
            coupling.isculled && break
            bond = coupling.bond
            i += 1

            U3_1 = cubic_U3_symmetrized_dipole(cubic_U31_dipole!, npt, bond, qs, qs_indices, (1, 1, 0))
            U3_2 = cubic_U3_symmetrized_dipole(cubic_U32_dipole!, npt, bond, qs, qs_indices, (0, 1, 1))
            U3_3 = cubic_U3_symmetrized_dipole(cubic_U31_dipole!, npt, bond, qs, qs_indices, (0, 0, 0))
            U3_4 = cubic_U3_symmetrized_dipole(cubic_U32_dipole!, npt, bond, qs, qs_indices, (0, 0, 0))
            U3_5 = cubic_U3_symmetrized_dipole(cubic_U31_dipole!, npt, bond, qs, qs_indices, (0, 0, 1))
            U3_6 = cubic_U3_symmetrized_dipole(cubic_U32_dipole!, npt, bond, qs, qs_indices, (1, 0, 0))
            U3_7 = cubic_U3_symmetrized_dipole(cubic_U31_dipole!, npt, bond, qs, qs_indices, (1, 1, 1))
            U3_8 = cubic_U3_symmetrized_dipole(cubic_U32_dipole!, npt, bond, qs, qs_indices, (1, 1, 1))

            V31 = real_space_cubic_vertices[i].V31
            V32 = real_space_cubic_vertices[i].V32

            @. U3 += V31 * (U3_1+0.25*U3_3) + conj(V31) * (U3_2+0.25*U3_4) + V32 * (U3_5+0.25*U3_7) + conj(V32) * (U3_6+0.25*U3_8)
        end
    end

    S = (sys.Ns[1]-1)/2
    for i in 1:L
        (; c2, c4) = stevens_coefs[i]
        c₂ = 1 - 1/(2S)
        c₄ = 1 - 3/S + 11/(4S^2) - 3/(4S^3)

        # No need to unrenormalize is the renormalization is zero
        c₂ = iszero(c₂) ? 1.0 : c₂
        c₄ = iszero(c₄) ? 1.0 : c₄

        c2_new = c2 ./ c₂
        c4_new = c4 ./ c₄

        U30_1 = cubic_U3′_symmetrized_dipole(cubic_U31′_dipole!, npt, qs_indices, i)
        U30_2 = cubic_U3′_symmetrized_dipole(cubic_U32′_dipole!, npt, qs_indices, i)
        U30_3 = cubic_U3′_symmetrized_dipole(cubic_U33′_dipole!, npt, qs_indices, i)
        U30_4 = cubic_U3′_symmetrized_dipole(cubic_U34′_dipole!, npt, qs_indices, i)

        @. U3 += 3*√(2S)/4 * ( (c2_new[2]+1im*c2_new[4]) * U30_1 + (c2_new[2]-1im*c2_new[4]) * U30_2 ) + 0.5 * (2S)^(3/2)*S * ( (c4_new[2]-1im*c4_new[8]) * U30_3 + (c4_new[2]+1im*c4_new[8]) * U30_4 )
    end

    return U3
end