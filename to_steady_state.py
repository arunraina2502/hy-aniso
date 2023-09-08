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


from hydiff import *
import numpy as np
import matplotlib.pyplot as plt
import pprint


class RunData:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run_solver(run_datum):
    pprint.pprint(run_datum.__dict__)
    pset = PSet()
    pset._N_T = run_datum.N_T_coeff*pset._N_L

    mesh = UnitSquareMesh(mpi_comm_self(), 32, 32)
    # mesh = UnitCubeMesh(mpi_comm_self(), 32, 32, 32)
    solver = GasMetalDiffusionSolver(mesh, pset)

    solver.update_angle(run_datum.angle)
    solver.update_beta(run_datum.beta)
    solver.update_gamma(run_datum.gamma)
    solver.update_temperature(run_datum.temp)
    t_lag = solver.solve(flux_limit=0.999, return_times=True, permitted_diff=0.01)

    run_datum.t_lag = t_lag
    run_datum.times = solver.get_times()
    run_datum.fluxes = solver.get_fluxes()
    run_datum.D11 = solver.get_D11()

    return run_datum


import time
time_str = time.strftime("%Y-%m-%d-%Hh-%Mm-%Ss")

if __name__ == "__main__":

    temp = 293.0
    N = 1e-3
    beta = 1.0
    gamma = 1.0
    angle = np.pi/6.0
    run_data = [RunData(percentage=100.0, angle=angle, beta=beta, gamma=gamma, N_T_coeff=N, temp=temp)]

    results = map(run_solver, run_data)
    datum = results[0]
    fluxes = datum.fluxes
    times = datum.times
    plt.semilogx(times, fluxes)
    plt.xlabel(r"$\bar{t}$")
    plt.ylabel(r"$\bar{D}_{11}$")
    plt.show()