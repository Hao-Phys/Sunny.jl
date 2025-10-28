mutable struct SelfConsistentNLSWT
    swt :: SpinWaveTheory
    real_space_quartic_vertices :: Vector{RealSpaceQuarticVerticesDipole}
    mean_field_values :: Vector{ComplexF64}
end

function SelfConsistentNLSWT(swt::SpinWaveTheory)
    @assert swt.sys.mode == :dipole "Self-consistent NLSWT only implemented for dipole mode"
    (; sys) = swt
    real_space_quartic_vertices = calculate_real_space_quartic_vertices_dipole(sys)
    num_interactions = length(real_space_quartic_vertices)
    mean_field_values = zeros(ComplexF64, 6num_interactions)

    return SelfConsistentNLSWT(swt, real_space_quartic_vertices, mean_field_values)
end

function Base.show(io::IO, ::MIME"text/plain", scnlswt::SelfConsistentNLSWT)
    (; swt, mean_field_values, real_space_quartic_vertices) = scnlswt
    printstyled(io, "SelfConsistentNLSWT ", mode_to_str(swt.sys), "\n"; bold=true, color=:underline)
    num_interactions = length(real_space_quartic_vertices)
    println("Number of interactions: ", num_interactions)
    println("Mean field values: ")
    for i in 1:num_interactions
        mfv = mean_field_values[6*(i-1)+1:6*(i-1)+6]
        println("Interaction ", i, ": ")
        println("Nii= ", mfv[1], " Njj= ", mfv[2], " Nij= ", mfv[3])
        println("Δii= ", mfv[4], " Δjj= ", mfv[5], " Δij= ", mfv[6])
    end
end

function swt_hamiltonian_dipole_nlsw!(H::Matrix{ComplexF64}, scnlswt::SelfConsistentNLSWT, q_reshaped::Vec3)
    (; swt, mean_field_values, real_space_quartic_vertices) = scnlswt
    (; sys) = swt
    (; sys, data) = swt
    (; stevens_coefs) = data

    L = nbands(swt)
    @assert size(H) == (2L, 2L)

    # Initialize Hamiltonian buffer 
    # Note that H11 for b†b, H22 for bb†, H12 for b†b†, and H21 for bb
    H .= 0.0 
    H11 = view(H, 1:L, 1:L)
    H12 = view(H, 1:L, L+1:2L)
    H21 = view(H, L+1:2L, 1:L)
    H22 = view(H, L+1:2L, L+1:2L)

    index = 0
    for (i, int) in enumerate(sys.interactions_union)

        # Single-ion anisotropy
        (; c2, c4, c6) = stevens_coefs[i]
        @assert iszero(c2) "Rank 2 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c4) "Rank 4 Stevens operators not supported in :dipole non-perturbative calculations yet"
        @assert iszero(c6) "Rank 6 Stevens operators not supported in :dipole non-perturbative calculations yet"

        # Pair interactions
        for coupling in int.pair
            (; isculled, bond) = coupling
            isculled && break

            index += 1
            Nii, Njj, Nij, Δii, Δjj, Δij = mean_field_values[6*(index-1)+1:6*(index-1)+6]
            (; V41, V42, V43) = real_space_quartic_vertices[index]

            (; i, j) = bond
            phase = exp(2π*im * dot(q_reshaped, bond.n)) # Phase associated with periodic wrapping

            # Bilinear exchange
            if !iszero(coupling.bilin)

                Q = V41 * conj(Nij) + V42 * Δii + conj(V42) * conj(Δjj) + 2 * conj(V43) * (Nii + Njj)

                H11[i, j] += Q * phase
                H11[j, i] += conj(Q) * conj(phase)
                H22[i, j] += conj(Q) * phase
                H22[j, i] += Q  * conj(phase)

                P = V41 * conj(Δij) + 2 * V42 * (Nii + Njj) + V43 * conj(Δjj) + conj(V43) * conj(Δii)

                H21[i, j] += P * phase
                H21[j, i] += P * conj(phase)
                H12[i, j] += conj(P) * phase
                H12[j, i] += conj(P) * conj(phase)

                Qi = V41 * Njj + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                Qj = V41 * Nii + 2 * V42 * Δij + 2 * conj(V42) * conj(Δij) + 2 * V43 * conj(Nij) + 2 * conj(V43) * Nij
                H11[i, i] += Qi
                H11[j, j] += Qj
                H22[i, i] += Qi
                H22[j, j] += Qj

                Pi = V42 * Nij + V43 * conj(Δij)
                Pj = V42 * conj(Nij) + conj(V43) * conj(Δij)
                H21[i, i] += Pi
                H21[j, j] += Pj
                H12[i, i] += conj(Pi)
                H12[j, j] += conj(Pj)
            end

            # Biquadratic exchange
            if !iszero(coupling.biquad)
                @error "Biquadratic exchange not supported in :dipole perturbative calculations yet"
            end
        end

    end
end

function update_mean_field_values!(scnlswt::SelfConsistentNLSWT, mean_field_values::Vector{ComplexF64})
    num_interactions = length(scnlswt.real_space_quartic_vertices)
    @assert length(mean_field_values) == 6num_interactions
    copyto!(scnlswt.mean_field_values, mean_field_values)
end

# We use multithreading to compute the integrand
function self_consistent_nlswt!(f, x, scnlswt::SelfConsistentNLSWT, hcubature_opts::NamedTuple=(;))
    (; swt) = scnlswt
    L = nbands(swt)

    update_mean_field_values!(scnlswt, x)

    (; sys) = swt
    # Precompute all bonds to be summed over
    tasks = []
    index = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            index += 1
            push!(tasks, (index, coupling))
        end
    end

    # Multithreading
    for i in eachindex(tasks)
        index, coupling = tasks[i]
        # Local buffers to avoid data racing
        H = zeros(ComplexF64, 2L, 2L)
        H_buf = zeros(ComplexF64, 2L, 2L)
        V = zeros(ComplexF64, 2L, 2L)
        mean_field_buf = zeros(ComplexF64, 6)

        bond = coupling.bond
        di, dj = bond.i, bond.j
        ret = hcubature((0,0,0), (1,1,1); hcubature_opts...) do k
            swt_hamiltonian_dipole!(H, swt, Vec3(k))
            swt_hamiltonian_dipole_nlsw!(H_buf, scnlswt, Vec3(k))
            @. H += H_buf
            bogoliubov!(V, H)

            mean_field_buf .= 0.0

            for band in 1:L
                mean_field_buf[1] += V[di+L, band] * conj(V[di+L, band])
                mean_field_buf[2] += V[dj+L, band] * conj(V[dj+L, band])
                mean_field_buf[3] += V[di+L, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
                mean_field_buf[4] += V[di, band] * conj(V[di+L, band])
                mean_field_buf[5] += V[dj, band] * conj(V[dj+L, band])
                mean_field_buf[6] += V[di, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
            end

            return SVector{6}(mean_field_buf)
        end

        f[6*(index-1)+1:6*(index-1)+6] = ret[1] - x[6*(index-1)+1:6*(index-1)+6]
    end
end

# Here the opts.. is for hcubature
# Enable multithreading
function calculate_mean_field_values_lswt(swt::SpinWaveTheory; opts...)
    (; sys) = swt
    L = nbands(swt)

    # Precompute all bonds to be summed over
    tasks = []
    index = 0
    for int in sys.interactions_union
        for coupling in int.pair
            coupling.isculled && break
            index += 1
            push!(tasks, (index, coupling))
        end
    end

    num_tasks = length(tasks)
    x = zeros(ComplexF64, 6num_tasks)

    # TODO: Implement multithreading. Investigate: `hcubature` does not seem to be thread-safe. 
    for i in eachindex(tasks)
        index, coupling = tasks[i]
        # Local buffers to avoid data racing
        H = zeros(ComplexF64, 2L, 2L)
        V = zeros(ComplexF64, 2L, 2L)
        mean_field_buf = zeros(ComplexF64, 6)
        bond = coupling.bond
        di, dj = bond.i, bond.j

        ret = hcubature((0,0,0), (1,1,1); opts...) do k

            swt_hamiltonian_dipole!(H, swt, Vec3(k))
            bogoliubov!(V, H)

            mean_field_buf .= 0.0

            for band in 1:L
                mean_field_buf[1] += V[di+L, band] * conj(V[di+L, band])
                mean_field_buf[2] += V[dj+L, band] * conj(V[dj+L, band])
                mean_field_buf[3] += V[di+L, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
                mean_field_buf[4] += V[di, band] * conj(V[di+L, band])
                mean_field_buf[5] += V[dj, band] * conj(V[dj+L, band])
                mean_field_buf[6] += V[di, band] * conj(V[dj+L, band]) * exp(-2π*im * dot(k, bond.n))
            end

            return SVector{6}(mean_field_buf)
        end


        x[6*(index-1)+1:6*(index-1)+6] = ret[1]
    end

    return x
end

# The opts should contain: 1. opts for hcubature 2. opts for nlsolve
function solve_self_consistent_nlswt!(scnlswt::SelfConsistentNLSWT; mean_field_values::Vector{ComplexF64} = ComplexF64[], hcubature_opts::NamedTuple=(;), nlsolve_opts::NamedTuple=(;))
    (; swt) = scnlswt
    (; real_space_quartic_vertices) = scnlswt
    if isempty(mean_field_values)
        num_interactions = length(real_space_quartic_vertices)
        x0 = calculate_mean_field_values_lswt(scnlswt.swt; hcubature_opts...)
    else
        num_interactions = length(real_space_quartic_vertices)
        @assert length(mean_field_values) == 6num_interactions
        x0 = copy(mean_field_values)
    end
    update_mean_field_values!(scnlswt, x0)
    sce_eqn! = (f, x) -> self_consistent_nlswt!(f, x, scnlswt, hcubature_opts)
    ret = nlsolve(sce_eqn!, x0; nlsolve_opts...)
    !converged(ret) && @warn "Self-consistent NLSWT converged to a solution with residual $(ret.residual_norm)"
    update_mean_field_values!(scnlswt, ret.zero)

    return ret.residual_norm
end

# TODO: Have a common interface with LSWT module for all code below
function excitations_scnlswt!(T, tmp1, tmp2, scnlswt::SelfConsistentNLSWT, q)
    (; swt) = scnlswt
    L = nbands(swt)
    size(T) == size(tmp1) == size(tmp2) == (2L, 2L) || error("Arguments T and tmp must be $(2L)×$(2L) matrices")

    q_reshaped = to_reshaped_rlu(swt.sys, q)
    dynamical_matrix!(tmp1, swt, q_reshaped)
    swt_hamiltonian_dipole_nlsw!(tmp2, scnlswt, q_reshaped)
    @. tmp1 += tmp2
    try
        return bogoliubov!(T, tmp1)
    catch _
        error("Instability at wavevector q = $q")
    end
end

function excitations_scnlswt(scnlswt::SelfConsistentNLSWT, q)
    @assert scnlswt.swt.sys.mode == :dipole ":SUN mode not supported yet"
    L = nbands(scnlswt.swt)
    T = zeros(ComplexF64, 2L, 2L)
    H_lsw  = zeros(ComplexF64, 2L, 2L)
    H_nlsw = zeros(ComplexF64, 2L, 2L)
    energies = excitations_scnlswt!(T, copy(H_lsw), copy(H_nlsw), scnlswt, q)
    return (energies, T)
end

function dispersion_scnlswt(scnlswt::SelfConsistentNLSWT, qpts)
    L = nbands(scnlswt.swt)
    qpts = convert(AbstractQPoints, qpts)
    disp = zeros(L, length(qpts.qs))
    for (iq, q) in enumerate(qpts.qs)
        view(disp, :, iq) .= view(excitations_scnlswt(scnlswt, q)[1], 1:L)
    end
    return reshape(disp, L, size(qpts.qs)...)
end

# TODO: Add the quantum corrections to the operators as well. At this moment, only the quantum corrections to the magnon eigenstates are included.
function intensities_bands_scnlswt(scnlswt::SelfConsistentNLSWT, qpts; kT=0, with_negative=false)
    (; swt) = scnlswt
    (; sys, measure) = swt
    isempty(measure.observables) && error("No observables! Construct SpinWaveTheory with a `measure` argument.")
    with_negative && error("Option `with_negative=true` not yet supported.")

    qpts = convert(AbstractQPoints, qpts)
    cryst = orig_crystal(sys)

    # Number of (magnetic) atoms in magnetic cell
    @assert sys.dims == (1,1,1)
    Na = nsites(sys)
    # Number of chemical cells in magnetic cell
    Ncells = Na / natoms(cryst)
    # Number of quasiparticle modes
    L = nbands(swt)
    # Number of wavevectors
    Nq = length(qpts.qs)

    # Temporary storage for pair correlations
    Nobs = num_observables(measure)
    Ncorr = num_correlations(measure)
    corrbuf = zeros(ComplexF64, Ncorr)

    # Preallocation
    T = zeros(ComplexF64, 2L, 2L)
    H = zeros(ComplexF64, 2L, 2L)
    H_nlswt = zeros(ComplexF64, 2L, 2L)
    Avec_pref = zeros(ComplexF64, Nobs, Na)
    disp = zeros(Float64, L, Nq)
    intensity = zeros(eltype(measure), L, Nq)


    for (iq, q) in enumerate(qpts.qs)
        q_global = cryst.recipvecs * q
        view(disp, :, iq) .= view(excitations_scnlswt!(T, H, H_nlswt, scnlswt, q), 1:L)

        for i in 1:Na, μ in 1:Nobs
            r_global = global_position(sys, (1,1,1,i)) # + offsets[μ,i]
            ff = get_swt_formfactor(measure, μ, i)
            Avec_pref[μ, i] = exp(- im * dot(q_global, r_global))
            Avec_pref[μ, i] *= compute_form_factor(ff, norm2(q_global))
        end

        Avec = zeros(ComplexF64, Nobs)

        # Fill `intensity` array
        for band in 1:L
            fill!(Avec, 0)
            if sys.mode == :SUN
                data = swt.data::SWTDataSUN
                N = sys.Ns[1]
                t = reshape(view(T, :, band), N-1, Na, 2)
                for i in 1:Na, μ in 1:Nobs
                    O = data.observables_localized[μ, i]
                    for α in 1:N-1
                        Avec[μ] += Avec_pref[μ, i] * (O[α, N] * t[α, i, 2] + O[N, α] * t[α, i, 1])
                    end
                end
            else
                @assert sys.mode in (:dipole, :dipole_uncorrected)
                data = swt.data::SWTDataDipole
                t = reshape(view(T, :, band), Na, 2)
                for i in 1:Na, μ in 1:Nobs
                    O = data.observables_localized[μ, i]
                    # This is the Avec of the two transverse and one
                    # longitudinal directions in the local frame. (In the
                    # local frame, z is longitudinal, and we are computing
                    # the transverse part only, so the last entry is zero)
                    displacement_local_frame = SA[t[i, 2] + t[i, 1], im * (t[i, 2] - t[i, 1]), 0.0]
                    Avec[μ] += Avec_pref[μ, i] * (data.sqrtS[i]/√2) * (O' * displacement_local_frame)[1]
                end
            end

            map!(corrbuf, measure.corr_pairs) do (α, β)
                Avec[α] * conj(Avec[β]) / Ncells
            end
            intensity[band, iq] = thermal_prefactor(disp[band]; kT) * measure.combiner(q_global, corrbuf)
        end
    end

    disp = reshape(disp, L, size(qpts.qs)...)
    intensity = reshape(intensity, L, size(qpts.qs)...)
    return BandIntensities(cryst, qpts, disp, intensity)
end

function intensities_scnlswt!(data, scnlswt::SelfConsistentNLSWT, qpts; energies, kernel::AbstractBroadening, kT=0)
    @assert size(data) == (length(energies), size(qpts.qs)...)
    bands = intensities_bands_scnlswt(scnlswt, qpts; kT)
    @assert eltype(bands) == eltype(data)
    broaden!(data, bands; energies, kernel)
    return Intensities(bands.crystal, bands.qpts, collect(Float64, energies), data)
end

function intensities_scnlswt(scnlswt::SelfConsistentNLSWT, qpts; energies, kernel::AbstractBroadening, kT=0)
    return broaden(intensities_bands_scnlswt(scnlswt, qpts; kT); energies, kernel)
end