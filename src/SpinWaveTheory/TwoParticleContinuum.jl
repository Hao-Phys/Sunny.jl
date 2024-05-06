"""
    dssf_tc(swt::SpinWaveTheory, q, ωs, η::Float64; opts...)

Given a [`SpinWaveTheory`](@ref) object, computes the dynamical spin structure
factor,
```math
    𝒮^{αβ}(𝐪, ω) = 1/(2πN)∫dt ∑_𝐫 \\exp[i(ωt - 𝐪⋅𝐫)] ⟨S^α(𝐫, t)S^β(0, 0)⟩,
```

from the two-particle continuum contribution,

```math
    𝒮^{αβ}(𝐪, ω) =1/N ∑_n ∑_m ∑_{𝐤}  |A_n^{αβ}_{nm}(𝐤, 𝐪)|^2 δ[ω-ω_{n}(𝐪+𝐤)-ω_{m}(-𝐤)].
```

``q`` must be a 3-vector in reciprocal lattice units (RLU), i.e., in the basis of
reciprocal lattice vectors.
"""
function dssf_tc(swt::SpinWaveTheory, q, ωs, η::Float64; opts...)
    (; sys, data, observables) = swt
    (; observable_operators) = data
    q_reshaped = to_reshaped_rlu(swt.sys, q)

    # Number of atoms in magnetic cell
    Nm = length(sys.dipoles)
    # Number of chemical cells in magnetic cell
    Ncells = Nm / natoms(orig_crystal(sys))
    # Dimension of Hilbert space
    N = sys.Ns[1]
    # Number of quasiparticle modes
    L = nbands(swt)

    # Get some integer numbers for later use
    numωs = length(ωs)
    num_obs = num_observables(observables)
    num_corrs = num_correlations(observables)

    # Preallocation
    # H1, V1 for q+k
    H1 = zeros(ComplexF64, 2L, 2L)
    V1 = zeros(ComplexF64, 2L, 2L)
    # H2, V2 for -k
    H2 = zeros(ComplexF64, 2L, 2L)
    V2 = zeros(ComplexF64, 2L, 2L)

    Avec_pref = zeros(ComplexF64, Nm)
    Avec = zeros(ComplexF64, num_obs, L, L)
    corrs_buf = zeros(ComplexF64, num_corrs*numωs)

    for i = 1:Nm
        @assert Nm == natoms(sys.crystal)
        Avec_pref[i] = exp(-2π*im * dot(q_reshaped, sys.crystal.positions[i]))
    end

    Sqω = hcubature((0,0,0), (1,1,1); opts...) do k_reshaped
        qpk_reshaped = q_reshaped + k_reshaped
        if sys.mode == :SUN
            swt_hamiltonian_SUN!(H1, swt, qpk_reshaped)
            swt_hamiltonian_SUN!(H2, swt, -k_reshaped)
        else
            @assert sys.mode in (:dipole, :dipole_large_S)
            swt_hamiltonian_dipole!(H1, swt, qpk_reshaped)
            swt_hamiltonian_dipole!(H2, swt, -k_reshaped)
        end

        disp1 = bogoliubov!(V1, H1)
        disp2 = bogoliubov!(V2, H2)

        # Fill the buffers with zeros
        Avec .= 0.0
        corrs_buf .= 0.0

        if sys.mode == :SUN
            for band1 = 1:L
                v1 = reshape(view(V1, :, band1), N-1, Nm, 2)
                for band2 = 1:L
                    v2 = reshape(view(V2, :, band2), N-1, Nm, 2)
                    for i = 1:Nm
                        for μ = 1:num_observables(observables)
                            @views O = observable_operators[:, :, μ, i]
                            for α = 1:N-1
                                for β = 1:N-1
                                    Avec[μ, band1, band2] += Avec_pref[i] * (O[α, β] - δ(α, β) * O[N, N]) * (v1[α, i, 2]*v2[β, i, 1] + v1[β, i, 1]*v2[α, i, 2])
                                end
                            end
                        end
                    end
                end
            end
        else
            @assert sys.mode in (:dipole, :dipole_large_S)
            for band1 = 1:L
                v1 = reshape(view(V1, :, band1), Nm, 2)
                for band2 = 1:L
                    v2 = reshape(view(V2, :, band2), Nm, 2)
                    for i = 1:Nm
                        for μ = 1:num_observables(observables)
                            @views O = observable_operators[:, :, μ, i]
                            Avec[μ, band1, band2] += Avec_pref[i] * O[1, 3] * (v1[i, 2]*v2[i, 1] + v1[i, 1]*v2[i, 2])
                        end
                    end
                end
            end
        end

        for (iω, ω) in enumerate(ωs)
            for (ci, i) in observables.correlations
                (α, β) = ci.I
                for band1 in 1:L, band2 in 1:L
                    corrs_buf[(iω-1)*num_corrs+i] += Avec[α, band1, band2] * conj(Avec[β, band1, band2]) * lorentzian(ω-disp1[band1]-disp2[band2], η) / Ncells
                end
            end
        end

        return SVector{num_corrs*numωs}(corrs_buf)
    end

    return reshape(Sqω[1], num_corrs, numωs)
end