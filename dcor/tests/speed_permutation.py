import numpy as np
import dcor
import sys


def main():
    resamples = int(sys.argv[1])
    num_samples = 100
    np.random.seed(0)
    a = np.random.normal(loc=1, size=(num_samples, 1))
    b = np.random.normal(loc=2, size=(num_samples, 1))
    result = dcor.homogeneity.energy_test(
        a,
        b,
        num_resamples=resamples
    )
    print(result.p_value)


if __name__ == '__main__':
    main()
