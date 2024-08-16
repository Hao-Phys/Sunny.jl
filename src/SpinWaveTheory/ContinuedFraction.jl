function continued_fraction_initial_states(npt::NonPerturbativeTheory, q, qcom_carts_index::CartesianIndex{3})
    (; swt, two_particle_states, clustersize) = npt
    (; sys, data, observables) = swt
    (; observable_operators) = data
    cryst = orig_crystal(sys)
    q_global = cryst.recipvecs * q

    num_obs = num_observables(observables)

    Nm = length(sys.dipoles)
    N  = sys.Ns[1]
    Nu = clustersize[1] * clustersize[2] * clustersize[3]
    S = (N-1) / 2
    sqrt_halfS = √(S/2)

    num_1ps = nbands(swt)
    num_2ps = length(two_particle_states[qcom_carts_index])

    f0s = zeros(ComplexF64, num_1ps+num_2ps, num_obs)
    # Note that: in Sunny.jl, the convention for the dynamical spin structure factor is normalized to the number of unit cells instead of the number of sites. As a result, we do not include the 1/√Nm factor in for the initial state.

    Avec_pref = zeros(ComplexF64, Nm)
    for i in 1:Nm
        r_global = global_position(sys, (1,1,1,i))
        Avec_pref[i] = exp(1im * dot(q_global, r_global))
    end
    
    Vq = npt.Vps[:, :, qcom_carts_index]
    if sys.mode == :SUN
        # Get the initial state components for the one-particle states
        for band in 1:num_1ps
            vq = reshape(view(Vq, :, band), N-1, Nm, 2)
            for i in 1:Nm
                for μ in 1:num_obs
                    @views O = observable_operators[:, :, μ, i]
                    for α in 1:N-1
                        f0s[band, μ] += Avec_pref[i] * (O[α, N] * conj(vq[α, i, 1]) + O[N, α] * conj(vq[α, i, 2]))
                    end
                end
            end
        end
        
        # Get the initial state components for the two-particle states
        for state in two_particle_states[qcom_carts_index]
            is = state.com_index + num_1ps

            q1_carts_index = state.q1_carts_index
            q2_carts_index = state.q2_carts_index

            vq1 = reshape(view(npt.Vps[:, :, q1_carts_index], :, state.band1), N-1, Nm, 2)
            vq2 = reshape(view(npt.Vps[:, :, q2_carts_index], :, state.band2), N-1, Nm, 2)

            for i in 1:Nm
                for μ in 1:num_obs
                    @views O = observable_operators[:, :, μ, i]
                    for α in 1:N-1
                        for β in 1:N-1
                            f0s[is, μ] += Avec_pref[i] * (O[α, β] - O[N, N] * δ(α, β)) * (conj(vq1[α, i, 1]) * conj(vq2[β, i, 2]) + conj(vq1[β, i, 2]) * conj(vq2[α, i, 1])) / (√Nu * state.ζ)
                        end
                    end
                end
            end

        end

    else
        # TODO: add a more clear note for this part, now I will follow Zhentao's note
        @assert sys.mode in (:dipole, :dipole_large_S)
        for band in 1:num_1ps
            vq = reshape(view(Vq, :, band), Nm, 2)
            for i in 1:Nm
                for μ in 1:num_obs
                    displacement_local_frame = conj([vq[i, 2] + vq[i, 1], im * (vq[i, 2] - vq[i, 1]), 0.0])
                    @views O_local_frame = observable_operators[:, :, μ, i]
                    f0s[band, μ] += Avec_pref[i] * sqrt_halfS * (O_local_frame * displacement_local_frame)[1]
                end
            end
        end

        for state in two_particle_states[qcom_carts_index]
            is = state.com_index + num_1ps

            q1_carts_index = state.q1_carts_index
            q2_carts_index = state.q2_carts_index

            vq1 = reshape(view(npt.Vps[:, :, q1_carts_index], :, state.band1), Nm, 2)
            vq2 = reshape(view(npt.Vps[:, :, q2_carts_index], :, state.band2), Nm, 2)

            for i in 1:Nm
                for μ in 1:num_obs
                    @views O = observable_operators[:, :, μ, i]
                    f0s[is, μ] += Avec_pref[i] * O[1, 3] * (conj(vq1[i, 2])*conj(vq2[i, 1]) + conj(vq1[i, 1])*conj(vq2[i, 2]) )  / (√Nu * state.ζ)
                end
            end

        end
    end

    return f0s
end

function modified_lanczos_aux!(as, bs, H, f0, niters)
    if norm(f0) < 1e-12
        as .= 0
        bs .= 0
    else
        f_curr = zeros(ComplexF64, length(f0))
        f_next = zeros(ComplexF64, length(f0))

        f_prev = copy(f0)
        mul!(f_next, H, f_prev)
        as[1] = real(dot(f_next, f_prev)) / real(dot(f_prev, f_prev))
        @. f_next = f_next - as[1] * f_prev

        for j in 2:niters
            @. f_curr = f_next
            bs[j-1] = real(dot(f_curr, f_curr))
            # If the norm of the current state is too small, we set the state to zero
            if abs(bs[j-1]) < 1e-12
                bs[j-1] = 0
                @. f_next = f_curr = 0
                as[j] = 0
            else
                bs[j-1] /= real(dot(f_prev, f_prev))
                mul!(f_next, H, f_curr)
                as[j] = real(dot(f_next, f_curr)) / real(dot(f_curr, f_curr))
                @. f_next = f_next - as[j] * f_curr - bs[j-1] * f_prev
                f_prev, f_curr = f_curr, f_prev
            end
        end
    end
end

function dssf_continued_fraction(npt::NonPerturbativeTheory, q, ωs, η::Float64, niters::Int; single_particle_correction::Bool=true, opts...)
    (; clustersize, swt) = npt
    Nu1, Nu2, Nu3 = clustersize
    all_qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    @assert q in all_qs "The momentum is not in the grid."

    # Here we mod one. This is because the q_reshaped is in the reciprocal lattice unit, and we need to find the closest q in the grid.
    q_reshaped = to_reshaped_rlu(npt.swt.sys, q)
    for i in 1:3
        (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
    end
    # Here we mod one. This is because the q_reshaped is in the reciprocal lattice unit, and we need to find the closest q in the grid.
    q_reshaped = mod.(q_reshaped, 1.0)
    qcom_carts_index = findmin(x -> norm(x - q_reshaped), all_qs)[2]

    # Calculate initial states for all observables
    f0s = continued_fraction_initial_states(npt, q, qcom_carts_index)

    num_1ps = nbands(npt.swt)
    num_2ps = length(npt.two_particle_states[qcom_carts_index])

    H = zeros(ComplexF64, num_1ps+num_2ps, num_1ps+num_2ps)
    H1ps = view(H, 1:num_1ps, 1:num_1ps)
    H2ps = view(H, num_1ps+1:num_1ps+num_2ps, num_1ps+1:num_1ps+num_2ps)
    H12ps = view(H, 1:num_1ps, num_1ps+1:num_1ps+num_2ps)
    one_particle_hamiltonian!(H1ps, npt, qcom_carts_index; single_particle_correction, opts...)
    two_particle_hamiltonian!(H2ps, npt, qcom_carts_index)
    one_to_two_particle_hamiltonian!(H12ps, npt, qcom_carts_index)

    # At this moment, we only support the correlation function for the same observable
    as = zeros(niters)
    bs = zeros(niters-1)

    num_obs = num_observables(swt.observables)
    ret = zeros(length(ωs), num_obs)
    for i in 1:num_obs
        f0 = view(f0s, :, i)
        modified_lanczos_aux!(as, bs, H, f0, niters)
        for (iω, ω) in enumerate(ωs)
            z = ω + 1im*η
            G = z - as[niters]
            for j in niters-1:-1:1
                A = bs[j] / G
                G = z - as[j] - A
            end
            G = inv(G) * real(dot(f0, f0))
            ret[iω, i] = - imag(G) / π
        end
    end

    return ret
end

function dssf_continued_fraction_two_particle(npt::NonPerturbativeTheory, q, ωs, η::Float64, niters::Int)
    (; clustersize, swt) = npt
    Nu1, Nu2, Nu3 = clustersize
    all_qs = [[i/Nu1, j/Nu2, k/Nu3] for i in 0:Nu1-1, j in 0:Nu2-1, k in 0:Nu3-1]
    @assert q in all_qs "The momentum is not in the grid."

    # the purpose of the code block below is to find the cartesian index of the q_reshpaed ∈ [0, 1)
    q_reshaped = to_reshaped_rlu(npt.swt.sys, q)
    for i in 1:3
        (abs(q_reshaped[i]) < 1e-12) && (q_reshaped = setindex(q_reshaped, 0.0, i))
    end
    # Here we mod one. This is because the q_reshaped is in the reciprocal lattice unit, and we need to find the closest q in the grid.
    q_reshaped = mod.(q_reshaped, 1.0)
    qcom_carts_index = findmin(x -> norm(x - q_reshaped), all_qs)[2]

    # Calculate initial states for all observables
    f0s = continued_fraction_initial_states(npt, q, qcom_carts_index)

    num_1ps = nbands(swt)
    num_2ps = length(npt.two_particle_states[qcom_carts_index])
    f0 = view(f0s, num_1ps+1:num_1ps+num_2ps, 3)

    H = zeros(ComplexF64, num_2ps, num_2ps)
    two_particle_hamiltonian!(H, npt, qcom_carts_index)

    # At this moment, we only support the correlation function for the same observable
    as = zeros(niters)
    bs = zeros(niters-1)

    ret = zeros(length(ωs))
    modified_lanczos_aux!(as, bs, H, f0, niters)
    for (iω, ω) in enumerate(ωs)
        z = ω + 1im*η
        G = z - as[niters]
        for j in niters-1:-1:1
            A = bs[j] / G
            G = z - as[j] - A
        end
        G = inv(G) * real(dot(f0, f0))
        ret[iω] = - imag(G) / π
    end

    return ret
end