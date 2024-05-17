import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def readmatrix(filename):
    '''
    Read a matrix from a csv file.
    '''
    return np.genfromtxt(filename, delimiter=',')

def create_adjacency_matrix(n):
    '''
    Create an adjacency matrix of a network.
    Inputs:
        n: number of nodes.
    Outputs:
        A: adjacency matrix of the network.
    '''
    
    G = nx.connected_watts_strogatz_graph(n, k=4, p=0.1, seed=42)
    A = nx.adjacency_matrix(G).toarray()
    return A

def forces(x, y, A, h):
    '''
    Compute the forces acting on the nodes of a network.
    Inputs:
        x: x-coordinates of the nodes.
        y: y-coordinates of the nodes.
        A: adjacency matrix of the network.
        h: parameter of the repulsion.
    Outputs:
        f_linked: forces acting on the nodes due to the links.
        f_unlinked: forces acting on the nodes due to the repulsion.
    '''
    N = len(x)
    xaux = np.outer(x, np.ones_like(x))
    yaux = np.outer(y, np.ones_like(y))
    dx = A * xaux - A * xaux.T
    dy = A * yaux - A * yaux.T
    dxy = np.sqrt(dx**2 + dy**2)

    Aind = np.where(A == 1)
    idiff = np.zeros((N, N))
    idiff[Aind] = 1 - 1.0 / dxy[Aind]
    fx = np.sum(idiff * dx, axis=1)
    fy = np.sum(idiff * dy, axis=1)
    f_linked = np.concatenate((fx, fy))

    A = np.ones_like(A) - A
    dx = A * xaux - A * xaux.T
    dy = A * yaux - A * yaux.T
    dxy = np.sqrt(dx**2 + dy**2)
    fac = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1 and i != j:
                fac[i, j] = min((1 - h / dxy[i, j]), 0)
    fx = np.sum(fac * dx, axis=1)
    fy = np.sum(fac * dy, axis=1)
    f_unlinked = np.concatenate((fx, fy))

    return f_linked + f_unlinked

def pot(x, y, A):
    '''
    Compute the potential energy of a network.
    Inputs:
        x: x-coordinates of the nodes.
        y: y-coordinates of the nodes.
        A: adjacency matrix of the network.
        Outputs:
        U: potential energy of the network.
    '''
    
    N = len(x)
    xaux = np.outer(x, np.ones_like(x))
    yaux = np.outer(y, np.ones_like(y))
    dx = A * xaux - A * xaux.T
    dy = A * yaux - A * yaux.T
    dxy = np.sqrt(dx**2 + dy**2)

    Aind = np.where(A == 1)
    I = np.zeros((N, N))
    I[Aind] = 1
    U1 = 0.5 * np.sum((dxy - I)**2)

    A = np.ones_like(A) - A
    dx = A * xaux - A * xaux.T
    dy = A * yaux - A * yaux.T
    dxy = np.sqrt(dx**2 + dy**2)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if A[i, j] == 1 and i != j:
                val = min(dxy[i, j] - np.sqrt(3), 0)
                J[i, j] = val**2
    U2 = 0.5 * np.sum(J)

    return U1 + U2

def plot_graph(G, pos, title, filename):
    '''
    Plot a graph.
    Inputs:
        G: graph to plot.
        pos: positions of the nodes.
        title: title of the plot.
        filename: name of the file where the plot is saved.
    '''
    
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='red', edge_color='blue', node_size=500, font_weight='bold')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.title(title)
    plt.savefig(filename)

def plot_force(force_values, iter, n, experiment_num, h):
    fsz = 20
    plt.figure(1)
    plt.clf()
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('k', fontsize=fsz)
    plt.ylabel(r'$||f(\mathbf{x},\mathbf{y})|| = ||-\nabla U(\mathbf{x},\mathbf{y})||$', fontsize=fsz)
    plt.plot(range(iter + 1), force_values[:iter + 1], linewidth=2)
    plt.title(f'Force Convergence for n={n} and h={h}')
    plt.savefig(f'results/force_plot_{experiment_num}.png')

def plot_energy(pot_values, iter, n, experiment_num, h):
    fsz = 20
    plt.figure(2)
    plt.clf()
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('k', fontsize=fsz)
    plt.ylabel(r'$U(\mathbf{x},\mathbf{y})$', fontsize=fsz)
    plt.plot(range(iter + 1), pot_values[:iter + 1], linewidth=2)
    plt.title(f'Energy Convergence for n={n} and h={h}')
    plt.savefig(f'results/energy_plot_{experiment_num}.png')
