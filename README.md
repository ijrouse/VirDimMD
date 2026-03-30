VirDimMD is a proof-of-concept molecular dynamics engine for running simulations in arbitrary dimensional spaces R^N rather than the typical R^3 space. 

Why bother? After all, the majority of systems we wish to simulate, and indeed ourselves, exist in R^3. So at first it may seem as allowing for extra dimensions is unnnecessary.
Consider, however, two walkers attempting to pass each other on a narrow pavement on their way home. In MD terms, this is a highly frustrated, metastable situation -- there is an obvious, more favourable state of existence, but since the walkers cannot walk through each other this state cannot be reached.
If, however, one walker temporarily steps off the pavement into the road, the other may pass by, and then both may return to the pavement and go home. Of course, the walker should not stay in the road too long, but this temporary displacement is enough to resolve the stalemate.

In MD terms, moving from R^1 to R^2 (stepping into the road) allowed the frustrated system to sidestep an energy barrier, after which we could return to R^1 to allow the system to reach its minimum energy configuration.

Let us now consider how this may apply to more realistic MD simulations. 
Let us assume we have generated an initial random state for a simulation in R^3, and this state is so dense that (almost) all particles are surrounded by other particles. 
To make a concrete example, we consider a mixture of two Lennard-Jones fluids, where each atom interacts with every other atom through the standard 6-12 potential:
U_{i,j} = 4 \epsilon_{i,j} ( (\sigma_{i,j}/r) ^12 - (\sigma_{i,j}/r)^6 )

We take highly unfavourable mixing:
\epsilon_{1,1} = \epsilon_{2,2} = 1
\epsilon_{1,2} = \epsilon_{2,1} = 0.1
and distribute these two liquids on a lattice with a high degree of mixing. This initial state is very high in energy compared to the phase-separated favourable state, but separation is kinetically slow due to the high density.
We now introduce an auxiliary dimension \alpha, boosting R^3 (x,y,z) to R^4 (x,y,z,\alpha) and assigning an additional position and velocity to each particle along the virtual axis. We ensure the LJ potential is computed accounting for this virtual axis,
i.e. r = \sqrt{ (x_i - x_j)^2 + (y_i - y_j)^2 + (z_i - z_j)^2 + (\alpha_i - \alpha_j)^2 }.
To ensure we remain close to realistic R^3 states, we add a harmonic potential 1/2 k_\alpha \alpha^2, with k_\alpha chosen to keep each atom within approximately $\sigma$ of $\alpha = 0$.
This allows particles to essentially slide past each other rather than having to cross a high thermal barrier by increasing $|\alpha_i - \alpha_j|$ while decreasing their "real space" distance. 
We then propagate the simulation for a fixed number of timesteps, or until some convergence criterion is reached, before slowly removing the virtual dimension to return the system to R^3 without introducing excess energy to the system.
The final result is a system defined entirely in R^3 and which may then be used as an improved initial state for production simulations, but which has the initial frustration greatly reduced. 

Internally, a distinction is made between "real" and "virtual" dimensions. At present, the sole difference is that virtual dimensions can be squeezed out using an automated routine to attempt to decrease all virtual co-ordinates to zero.
Note that this code is not intended to be highly optimised or for use in production simulations, but is instead intended to demonstrate the benefit of applying high-dimensional evolution during equilibriation of more standard MD simulations.
