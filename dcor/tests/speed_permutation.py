import numpy as np
import dcor

num_samples = 100
np.random.seed(0)
a = np.random.normal(loc=1, size=(num_samples, 1))
b = np.random.normal(loc=2, size=(num_samples, 1))
result = dcor.homogeneity.energy_test(
    a,
    b,
    num_resamples=5000
)
print(result.p_value)

