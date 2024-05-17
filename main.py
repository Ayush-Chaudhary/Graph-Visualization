from optimization import run_experiment
import numpy as np

if __name__ == '__main__':
    experiments = [
        (20, np.sqrt(3)),
        (30, np.sqrt(3)),
        (40, np.sqrt(3)),
        (50, np.sqrt(3)),
        (50, 5)
    ]

    for i, (n, h) in enumerate(experiments):
        run_experiment(n, h, i+1)