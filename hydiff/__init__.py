# Copyright (C) 2017 Nathan Sime and Arun Raina
#
# This file is part of the code supporting the paper, "Effect of anisotropy and
# regime of diffusion on the measurement of lattice diffusion coefficient of
# hydrogen in metals".
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this code. If not, see <http://www.gnu.org/licenses/>.


from dolfin import *
import multiprocessing as mp


def report(*args, **kwargs):
    print(str(str(mp.current_process().name) + ":\t"+ ", ".join(map(str, args))))


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Problem(NonlinearProblem):
    def __init__(self, a, L, bcs):
        NonlinearProblem.__init__(self)
        self.a = a
        self.L = L
        self.bcs = bcs

    def F(self, b, x):
        assembler = SystemAssembler(self.a, self.L, self.bcs)
        assembler.assemble(b, x)

    def J(self, A, x):
        assembler = SystemAssembler(self.a, self.L, self.bcs)
        assembler.assemble(A)


class CustomSolverDamped(NewtonSolver):
    def __init__(self, comm, mesh):
        self.mesh = mesh
        self.solver = PETScKrylovSolver(comm)
        NewtonSolver.__init__(self, comm, self.solver, PETScFactory.instance())
        self.rebuild = True

    def rebuild_pc_next_solve(self):
        self.rebuild = True

    def solver_setup(self, A, P, problem, iteration):
        self.solver.set_operator(A)
        PETScOptions.set("ksp_type", "cg")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)
        PETScOptions.set("pc_hypre_boomeramg_agg_num_paths", 2)
        PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.9)
        PETScOptions.set("pc_hypre_boomeramg_P_max", 5)
        PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)
        self.solver.set_from_options()
        self.set_relaxation_parameter(1.0)

    def update_solution(self, x, dx, relaxation_parameter, nonlinear_problem, iteration):
        tau = 10.0
        theta = min(sqrt(2.0*tau/norm(dx, norm_type="l2", mesh=self.mesh)), 1.0)
        info('update: %.5e' % theta)
        x.axpy(-theta, dx)


class CustomSolverUndamped(NewtonSolver):
    def __init__(self, comm, mesh):
        self.mesh = mesh
        self.solver = PETScKrylovSolver(comm)
        NewtonSolver.__init__(self, comm, self.solver, PETScFactory.instance())
        self.rebuild = True

    def rebuild_pc_next_solve(self):
        self.rebuild = True

    def solver_setup(self, A, P, problem, iteration):
        self.solver.set_operator(A)
        PETScOptions.set("ksp_type", "cg")
        PETScOptions.set("pc_type", "hypre")
        PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)
        PETScOptions.set("pc_hypre_boomeramg_agg_num_paths", 2)
        PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.9)
        PETScOptions.set("pc_hypre_boomeramg_P_max", 5)
        PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)

        self.solver.set_from_options()
        self.set_relaxation_parameter(1.0)


class PSet:

    # Material Constants
    _Q = 6.7e3
    _D_0 = 1e-7
    _R = 8.3144
    _T_0 = Constant(293.0)
    _L = 5e-3
    _N_L = 8.46e28
    _N_T = 1e-3*_N_L
    _dH = -35e3
    _theta_L_0 = 1e-6


class GasMetalDiffusionSolver:

    def __init__(self, mesh, pset):
        # Dimensionless Coefficients
        Q = pset._Q/(pset._R*pset._T_0)
        N = Constant(pset._N_T/pset._N_L)
        dH = pset._dH/(pset._R*pset._T_0)

        K = exp(-dH)
        theta_L_0 = Constant(1e-6)

        V = FunctionSpace(mesh, "CG", 1)
        report("Dofs: " + str(V.dim()))

        # Test and approximation functions
        v = TestFunction(V)
        theta_L = Function(V)

        # Euler theta scheme. 0.5 = Crank Nicholson. 1.0 = Backward Euler.
        theta_L_m = interpolate(Constant(0.0), V)
        theta_L.vector()[:] = theta_L_m.vector()

        # theta_L mid time step
        et = 0.5
        theta_L_t = et*theta_L + (1 - et)*theta_L_m

        # Choose initial timestep based on problem type
        self.is_N_0 = abs(float(N)) < 1e-9
        self.dt = Constant(1e-3 if self.is_N_0 else 1.0)

        # Default system parameters. These can be assigned prior to system solve
        beta = Constant(1.0)
        gamma = Constant(1.0)
        angle = Constant(pi/3.0)

        # Initialise diffusion tensor based on problem dimension
        if mesh.geometry().dim() == 1:
            D_tilde = pset._D_0*exp(-pset.Q)
            D = D_tilde/pset._D_0

        elif mesh.geometry().dim() == 2:
            R = as_matrix(((cos(angle), -sin(angle)),
                           (sin(angle),  cos(angle))))
            D_tilde = as_matrix(((pset._D_0*exp(-Q),                     0),
                                 (                0, beta*pset._D_0*exp(-Q))))
            D = R*D_tilde*R.T/pset._D_0

        else:
            R = as_matrix(((         cos(angle)**2 - cos(angle)*sin(angle)**2, cos(angle)*sin(angle) + cos(angle)**2*sin(angle),         sin(angle)**2),
                           (-cos(angle)*sin(angle) - cos(angle)**2*sin(angle),                   -sin(angle)**2 + cos(angle)**3, cos(angle)*sin(angle)),
                           (                                    sin(angle)**2,                           -cos(angle)*sin(angle),            cos(angle))))
            D_tilde = as_matrix(((pset._D_0*exp(-Q),                      0,                      0),
                                 (                0, beta*pset._D_0*exp(-Q),                      0),
                                 (                0,                      0, gamma*pset._D_0*exp(-Q))))
            D = R*D_tilde*R.T/pset._D_0

        # Time dependent term
        F = (theta_L - theta_L_m)*(1 + N*K/(1 + K*theta_L_t*theta_L_0)**2)*v*dx

        # System essential boundary conditions
        bcs = [DirichletBC(V, Constant(1.0), 'near(x[0], 0.0)'),
               DirichletBC(V, Constant(0.0), 'near(x[0], 1.0)')]

        # Mark a facet function to compute the steady state flux
        ff = FacetFunction("size_t", mesh, 0)
        CompiledSubDomain("near(x[0], 1.0)").mark(ff, 1)
        CompiledSubDomain("near(x[0], 0.0)").mark(ff, 2)
        ds_f = Measure("ds", domain=mesh, subdomain_data=ff)

        # Formulate the diffusion term.
        F += self.dt*dot(D*grad(theta_L_t), grad(v))*dx

        # Compute the Frechet derivative
        J = derivative(F, theta_L)

        # Define the steady state solution.
        self.steady_soln = Function(V)
        self.steady_F = dot(D*grad(self.steady_soln), grad(v))*dx
        self.steady_flux = (-D*grad(self.steady_soln))[0]*ds_f(1)

        # Measured flux
        J_xx = (-D*grad(theta_L_m))[0]*ds_f(1)

        # Assign all members
        self.D = D
        self.bcs = bcs
        self.J_xx = J_xx
        self.J_xx_L = (-D*grad(theta_L))[0]*ds_f(1)
        self.theta_L = theta_L
        self.theta_L_m = theta_L_m
        self.V = V
        self.mesh = mesh
        self.J = J
        self.F = F
        self.beta = beta
        self.gamma = gamma
        self.angle = angle
        self.pset = pset

    def update_beta(self, v):
        self.beta.assign(v)

    def update_gamma(self, v):
        self.gamma.assign(v)

    def update_angle(self, v):
        self.angle.assign(v)

    def update_temperature(self, v):
        self.pset._T_0.assign(v)

    def get_D11(self):
        return float(self.D[0, 0])

    def get_times(self):
        return self.times

    def get_fluxes(self):
        return self.fluxes

    def solve(self, flux_limit=0.632, return_times=False, permitted_diff=0.1):
        # Solve the steady problem
        solve(self.steady_F == 0, self.steady_soln, self.bcs)

        # Set the initial time step based on the system temperature
        initial_dt = 1e-6 if float(self.pset._T_0) > 290.0 else 1e-6
        if float(self.pset._T_0) < 290 and near(self.pset._N_T, 0.0):
            initial_dt = 1e-6
        self.dt.assign(initial_dt)

        # Compute the steady state flux
        J_ss = assemble(self.steady_flux)

        # Apply initial conditions
        self.theta_L_m.vector()[:] = 0.0
        self.theta_L.vector()[:] = 0.0
        self.theta_L_m.interpolate(Expression("near(x[0], 0.0) ? 1.0 : 0.0", degree=1))

        # Results lists
        self.times = []
        self.fluxes = []

        t = 0.0
        self.times.append(0.0)
        self.fluxes.append(0.0)

        # Use a damped or undamped solver if the problem is nonlinear or linear, respectively
        problem = Problem(self.J, self.F, self.bcs)
        if near(self.pset._N_T, 0.0):
            solver = CustomSolverUndamped(self.mesh.mpi_comm(), self.mesh)
        else:
            solver = CustomSolverDamped(self.mesh.mpi_comm(), self.mesh)
        solver.parameters["absolute_tolerance"] = 1e-9
        solver.parameters["maximum_iterations"] = 100
        solver.parameters["error_on_nonconvergence"] = False

        # Commence main solution loop
        found_t_lag = False
        t_lag = 0.0
        j = 0
        old_flux_r = 0.0

        while j < 1000:  # Maximum of 1000 steps
            # Solve the time step
            solver.solve(problem, self.theta_L.vector())

            # Have we exceeded the flux limit? Resolve at greater fidelity
            while assemble(self.J_xx_L)/J_ss > flux_limit:
                self.dt.assign(float(self.dt)/2.0)
                report("ADAPT: dt_val", float(self.dt))
                solver.solve(problem, self.theta_L.vector())

            # Instruct the solver to rebuild the preconditioner
            solver.rebuild_pc_next_solve()

            # Update the solver time
            t += float(self.dt)

            # Adapt and limit the new time step
            diff_vec = (self.theta_L.vector() - self.theta_L_m.vector());
            diff_vec.abs()
            diff = diff_vec.max()

            dt_val = 0.97*float(self.dt)*(permitted_diff/diff)

            # Set the old time step as the new
            self.theta_L_m.vector()[:] = self.theta_L.vector()

            self.times.append(t)

            # Compute and store the flux at this time step
            flux = assemble(self.J_xx)
            self.fluxes.append(flux)

            # Assign the new time step within these limits
            dt_min = 1e-5
            dt_max = 1e5

            self.dt.assign(min(max(dt_val, dt_min), dt_max))

            report("Status: deviation of flux", abs(flux/J_ss - flux_limit), "dt: ", float(self.dt))

            # t_lag computed, return.
            if not found_t_lag and abs(flux/J_ss - flux_limit) < 1e-3:
                t_lag = t
                report("TLAG:", t_lag)
                break

            j += 1
            if j > 1000:
                raise Exception("Exceeded maximum solves")

        return t_lag
