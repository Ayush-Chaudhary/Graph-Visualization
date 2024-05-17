from optimization import run_experiment
import numpy as np

if __name__ == '__main__':
    experiments = [
        (10, np.sqrt(3)),
        (40, np.sqrt(3)),
        (10, 4),
        (40, 4),
        (15, 2)
    ]

    for i, (n, h) in enumerate(experiments):
        run_experiment(n, h, i+1)