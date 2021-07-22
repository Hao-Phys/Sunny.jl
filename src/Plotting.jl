"""Plotting functions for lattices and spins on lattices.
"""

import GLMakie

function plot_lattice(lattice::Lattice{2}; color=:blue, markersize=20, linecolor=:grey, linewidth=1.0, kwargs...)
    f = GLMakie.Figure()
    ax = GLMakie.Axis(f[1, 1])
    ax.autolimitaspect = 1
    GLMakie.hidespines!(ax)
    GLMakie.hidedecorations!(ax)

    # Plot the unit cell mesh
    plot_cells!(lattice; color=linecolor, linewidth=linewidth)

    # Plot markers at each site
    sites = reinterpret(reshape, Float64, collect(lattice))
    xs, ys = vec(sites[1, 1:end, 1:end, 1:end]), vec(sites[2, 1, 1:end, 1:end])
    GLMakie.scatter!(xs, ys; color=color, markersize=markersize, kwargs...)
    f
end

# 3D is a bit wonky at the moment - Axis3 doesn't seem to work with scatter!
# For now, have to plot sites below the unit cell grid
function plot_lattice(lattice::Lattice{3}; color=:blue, markersize=100, linecolor=:grey, linewidth=1.0, kwargs...)
    # f = Figure()
    # ax = Axis3(f[1, 1], viewmode=:fit)
    # hidespines!(ax)
    # hidedecorations!(ax)

    # Plot markers at each site
    sites = reinterpret(reshape, Float64, collect(lattice))
    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    f = GLMakie.scatter(xs, ys, zs; color=color, markersize=markersize, show_axis=false, kwargs...)

    # For some odd reason, the sites will not appear unless this happens afterwards
    # Plot the unit cell mesh
    plot_cells!(lattice; color=linecolor, linewidth=linewidth)
    f
end

# TODO: Base.Cartesian could combine these functions
"Plot the outlines of the unit cells of a lattice"
function plot_cells!(lattice::Lattice{2}; color=:grey, linewidth=1.0, kwargs...)
    lattice = brav_lattice(lattice)

    xs, ys = Vector{Float64}(), Vector{Float64}()
    nx, ny = lattice.size
    for j in 1:ny
        left_pt, right_pt = lattice[1, 1, j], lattice[1, nx, j]
        push!(xs, left_pt[1])
        push!(xs, right_pt[1])
        push!(ys, left_pt[2])
        push!(ys, right_pt[2])
    end
    for i in 1:nx
        bot_pt, top_pt = lattice[1, i, 1], lattice[1, i, ny]
        push!(xs, bot_pt[1])
        push!(xs, top_pt[1])
        push!(ys, bot_pt[2])
        push!(ys, top_pt[2])
    end

    GLMakie.linesegments!(xs, ys; color=color, linewidth=linewidth)
end

"Plot the outlines of the unit cells of a lattice"
function plot_cells!(lattice::Lattice{3}; color=:grey, linewidth=1.0, kwargs...)
    lattice = brav_lattice(lattice)

    xs, ys, zs = Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
    nx, ny, nz = lattice.size
    for j in 1:ny
        for k in 1:nz
            bot_pt, top_pt = lattice[1, 1, j, k], lattice[1, nx, j, k]
            push!(xs, bot_pt[1])
            push!(xs, top_pt[1])
            push!(ys, bot_pt[2])
            push!(ys, top_pt[2])
            push!(zs, bot_pt[3])
            push!(zs, top_pt[3])
        end
        for i in 1:nx
            left_pt, right_pt = lattice[1, i, j, 1], lattice[1, i, j, nz]
            push!(xs, left_pt[1])
            push!(xs, right_pt[1])
            push!(ys, left_pt[2])
            push!(ys, right_pt[2])
            push!(zs, left_pt[3])
            push!(zs, right_pt[3])
        end
    end
    for k in 1:nz
        for i in 1:nx
            left_pt, right_pt = lattice[1, i, 1, k], lattice[1, i, ny, k]
            push!(xs, left_pt[1])
            push!(xs, right_pt[1])
            push!(ys, left_pt[2])
            push!(ys, right_pt[2])
            push!(zs, left_pt[3])
            push!(zs, right_pt[3])
        end
    end

    GLMakie.linesegments!(xs, ys, zs; color=color, linewidth=linewidth)
end

function plot_spins(sys::SpinSystem{2}; linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.3, kwargs...)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.1 .* reinterpret(reshape, Float64, collect(sys.sites))

    xs = vec(sites[1, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end])
    zs = zero(xs)
    us = vec(spins[1, 1:end, 1:end, 1:end])
    vs = vec(spins[2, 1:end, 1:end, 1:end])
    ws = vec(spins[3, 1:end, 1:end, 1:end])

    GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
end

function plot_spins(sys::SpinSystem{3}; linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.3, kwargs...)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.2 .* reinterpret(reshape, Float64, collect(sys.sites))

    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    us = vec(spins[1, 1:end, 1:end, 1:end, 1:end])
    vs = vec(spins[2, 1:end, 1:end, 1:end, 1:end])
    ws = vec(spins[3, 1:end, 1:end, 1:end, 1:end])

    GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
end

"Equivalent to above, but different arguments"
function plot_spins(lat::Lattice{3}, spins; linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.3, kwargs...)
    sites = reinterpret(reshape, Float64, collect(lat))
    spins = reinterpret(reshape, Float64, spins.val)

    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    us = vec(spins[1, 1:end, 1:end, 1:end, 1:end])
    vs = vec(spins[2, 1:end, 1:end, 1:end, 1:end])
    ws = vec(spins[3, 1:end, 1:end, 1:end, 1:end])

    GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
end

# No support for higher than 3D visualization, sorry!

"Produce an animation of spin dynamics for the specified length of time"
function anim_integration(
    sys::SpinSystem{2}, fname, steps_per_frame, Δt, nframes;
    linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.2, kwargs...
)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
    
    xs = vec(sites[1, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end])
    zs = zero(ys)
    us = GLMakie.Node(vec(spins[1, 1:end, 1:end, 1:end]))
    vs = GLMakie.Node(vec(spins[2, 1:end, 1:end, 1:end]))
    ws = GLMakie.Node(vec(spins[3, 1:end, 1:end, 1:end]))
    fig, ax, plot = GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
    display(fig)

    framerate = 30
    integrator = SpinHeunP(sys)

    GLMakie.record(fig, fname, 1:nframes; framerate=framerate) do frame
        for step in 1:steps_per_frame
            evolve!(integrator, Δt)
        end
        spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
        us[] = vec(spins[1, 1:end, 1:end, 1:end])
        vs[] = vec(spins[2, 1:end, 1:end, 1:end])
        ws[] = vec(spins[3, 1:end, 1:end, 1:end])
    end
end

"Produce an animation of spin dynamics for the specified length of time"
function anim_integration(
    sys::SpinSystem{3}, fname, steps_per_frame, Δt, nframes;
    linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.2, kwargs...
)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
    
    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    us = GLMakie.Node(vec(spins[1, 1:end, 1:end, 1:end, 1:end]))
    vs = GLMakie.Node(vec(spins[2, 1:end, 1:end, 1:end, 1:end]))
    ws = GLMakie.Node(vec(spins[3, 1:end, 1:end, 1:end, 1:end]))
    fig, ax, plot = GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
    display(fig)

    framerate = 30
    integrator = SpinHeunP(sys)

    GLMakie.record(fig, fname, 1:nframes; framerate=framerate) do frame
        for step in 1:steps_per_frame
            evolve!(integrator, Δt)
        end
        spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
        us[] = vec(spins[1, 1:end, 1:end, 1:end, 1:end])
        vs[] = vec(spins[2, 1:end, 1:end, 1:end, 1:end])
        ws[] = vec(spins[3, 1:end, 1:end, 1:end, 1:end])
    end
end

"Endless integration in a live window"
function live_integration(
    sys::SpinSystem{2}, steps_per_frame, Δt;
    linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.2, kwargs...
)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
    
    xs = vec(sites[1, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end])
    zs = zero(ys)
    us = GLMakie.Node(vec(spins[1, 1:end, 1:end, 1:end]))
    vs = GLMakie.Node(vec(spins[2, 1:end, 1:end, 1:end]))
    ws = GLMakie.Node(vec(spins[3, 1:end, 1:end, 1:end]))
    fig, ax, plot = GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
    display(fig)

    framerate = 30
    integrator = SpinHeunP(sys)

    while true
        for step in 1:steps_per_frame
            evolve!(integrator, Δt)
        end
        spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
        us[] = vec(spins[1, 1:end, 1:end, 1:end])
        vs[] = vec(spins[2, 1:end, 1:end, 1:end])
        ws[] = vec(spins[3, 1:end, 1:end, 1:end])
        sleep(1/framerate)
    end
end

"Endless integration in a live window"
function live_integration(
    sys::SpinSystem{3}, steps_per_frame, Δt;
    linecolor=:grey, arrowcolor=:red, linewidth=0.1, arrowsize=0.2, kwargs...
)
    sites = reinterpret(reshape, Float64, collect(sys.lattice))
    spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
    
    xs = vec(sites[1, 1:end, 1:end, 1:end, 1:end])
    ys = vec(sites[2, 1:end, 1:end, 1:end, 1:end])
    zs = vec(sites[3, 1:end, 1:end, 1:end, 1:end])
    us = GLMakie.Node(vec(spins[1, 1:end, 1:end, 1:end, 1:end]))
    vs = GLMakie.Node(vec(spins[2, 1:end, 1:end, 1:end, 1:end]))
    ws = GLMakie.Node(vec(spins[3, 1:end, 1:end, 1:end, 1:end]))
    fig, ax, plot = GLMakie.arrows(
        xs, ys, zs, us, vs, ws;
        linecolor=linecolor, arrowcolor=arrowcolor, linewidth=linewidth, arrowsize=arrowsize,
        show_axis=false, kwargs...    
    )
    display(fig)

    framerate = 30
    integrator = SpinHeunP(sys)

    while true
        for step in 1:steps_per_frame
            evolve!(integrator, Δt)
        end
        spins = 0.2 .* reinterpret(reshape, Float64, sys.sites)
        us[] = vec(spins[1, 1:end, 1:end, 1:end, 1:end])
        vs[] = vec(spins[2, 1:end, 1:end, 1:end, 1:end])
        ws[] = vec(spins[3, 1:end, 1:end, 1:end, 1:end])
        sleep(1/framerate)
    end
end