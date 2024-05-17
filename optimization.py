import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from functions import forces, pot, plot_graph, plot_force, plot_energy, create_adjacency_matrix

def optimize(n, A, experiment_num, h):
    '''
    Optimize the positions of the nodes in the graph.
    Inputs:
        n: number of nodes in the graph.
        A: adjacency matrix of the graph.
        experiment_num: number of the experiment.
        h: parameter of the repulsion.
    '''

    N = A.shape[0]

    tol = 1e-12
    iter_max = 200
    Delta_max = 20
    Delta_min = 1e-15
    Delta = 10
    eta = 0.001
    rho_good = 0.8
    rho_bad = 0.2

    v = np.random.randn(2 * N) * N

    force_values = np.zeros(iter_max + 1)
    pot_values = np.zeros(iter_max + 1)
    g = forces(v[:N], v[N:], A, h)
    f = pot(v[:N], v[N:], A)
    force = np.linalg.norm(g)
    force_values[0] = force
    pot_values[0] = f

    B = np.eye(len(v))
    reset_length = 10
    iter = 1

    while force > tol and iter < iter_max:
        flag_boundary = 0

        pB = -np.linalg.solve(B, g)
        if np.linalg.norm(pB) < Delta:
            p = pB
        else:
            c = - (g.T @ g) / (g.T @ B @ g)
            pU = c * g
            if np.linalg.norm(pU) > Delta:
                p = (Delta * pU) / np.linalg.norm(pU)
            else:
                tau = (np.sqrt((pU.T @ (pB - pU))**2 + np.linalg.norm(pB - pU)**2 * (Delta**2 - np.linalg.norm(pU)**2)) - pU.T @ (pB - pU)) / np.linalg.norm(pB - pU)**2
                p = pU + tau * (pB - pU)
            flag_boundary = 1

        v_new = v + p
        f_new = pot(v_new[:N], v_new[N:], A)
        g_new = forces(v_new[:N], v_new[N:], A, h)
        m_new = f + g.T @ p + 0.5 * p.T @ B @ p
        rho = (f - f_new + 1e-14) / (f - m_new + 1e-14)
        if rho < rho_bad:
            Delta = max(0.25 * Delta, Delta_min)
        elif rho > rho_good and flag_boundary == 1:
            Delta = min(2 * Delta, Delta_max)

        if rho > eta:
            s = p
            y = g_new - g
            if (iter + 1) % reset_length == 0:
                B = np.eye(len(v))
            else:
                B = B + np.outer(y, y) / (y.T @ s) - (B @ np.outer(s, s) @ B) / (s.T @ B @ s)
            v = v_new
            f = f_new
            g = g_new
            force = np.linalg.norm(g)
        #     print(f'Accept: iter # {iter}: f = {f:.10f}, |df| = {force:.4e}')
        # else:
        #     print(f'Reject: iter # {iter}: f = {f:.10f}, |df| = {force:.4e}')

        iter += 1
        pot_values[iter] = f
        force_values[iter] = force

    plot_force(force_values, iter, n, experiment_num,h)

    plot_energy(pot_values, iter, n, experiment_num, h)

    # Use NetworkX to plot the graph with a force-directed layout
    G_optimized = nx.from_numpy_array(A)
    pos_optimized = {i: (v[i], v[N+i]) for i in range(N)}  # Updated positions from optimization
    plot_graph(G_optimized, pos_optimized, title=f'Optimized Graph for n={n}', filename=f'results/optimized_graph_{experiment_num}.png')

def run_experiment(n, h, experiment_num):
    '''
    Run an experiment with the given parameters.
    Inputs:
        n: number of nodes in the graph.
        h: parameter of the repulsion.
        experiment_num: number of the experiment.
    '''
    print(f"Running experiment {experiment_num} with n={n}, h={h}")

    # Generate a random connected adjacency matrix
    A = create_adjacency_matrix(n)

    # Plot and save the initial random graph
    G_random = nx.from_numpy_array(A)
    pos_random = nx.spring_layout(G_random, seed=42)
    plot_graph(G_random, pos_random, title=f'Random Graph for n={n}, h={h}', filename=f'results/random_graph_{experiment_num}.png')

    # Run the Q2 optimization function
    optimize(n, A, experiment_num, h)