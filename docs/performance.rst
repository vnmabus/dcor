Performance
===========

This section shows the relative performance of different algorithms used to
compute the functionalities offered in this package.

Distance covariance and distance correlation
--------------------------------------------

We offer currently three different algorithms with different performance
characteristics that can be used to compute the estimators of the squared
distance covariance, both biased and unbiased.

These algorithms are:

- The original naive algorithm, with order :math:`O(n^2)`.
- The AVL-inspired fast algorithm described in
  :cite:`d-fast_distance_correlation_avl`, which improved the performance and
  attained :math:`O(n\log n)` complexity.
- The mergesort algorithm described in
  :cite:`d-fast_distance_correlation_mergesort`, which also obtained
  :math:`O(n\log n)` complexity.

The following code shows the differences in performance executing the
algorithm 100 times for samples of different sizes. It then plots the
resulting graph.

.. jupyter-execute::

        import dcor
        import numpy as np
        from timeit import timeit
        import matplotlib.pyplot as plt
        
        np.random.seed(0)
        n_times = 100
        n_samples_list = [10, 50, 100, 500, 1000]
        avl_times = np.zeros(len(n_samples_list))
        mergesort_times = np.zeros(len(n_samples_list))
        naive_times = np.zeros(len(n_samples_list))
        
        for i, n_samples in enumerate(n_samples_list):
        	x = np.random.normal(size=n_samples)
        	y = np.random.normal(size=n_samples)
        		
        	def avl():
        		return dcor.distance_covariance(x, y, method='AVL')
        		
        	def mergesort():
        		return dcor.distance_covariance(x, y, method='MERGESORT')
        		
        	def naive():
        		return dcor.distance_covariance(x, y, method='NAIVE')
        		
        	if i == 0:
        		# Execute avl and mergesort at least once, so that
        		# Numba compilation times are not included in the comparison
        		avl()
        		mergesort()
        		
        	avl_times[i] = timeit(avl, number=n_times)
        	mergesort_times[i] = timeit(mergesort, number=n_times)
        	naive_times[i] = timeit(naive, number=n_times)
        
        plt.title("Distance covariance performance comparison")
        plt.xlabel("Number of samples")
        plt.ylabel("Time (seconds)")
        plt.plot(n_samples_list, avl_times, label="avl")
        plt.plot(n_samples_list, mergesort_times, label="mergesort")
        plt.plot(n_samples_list, naive_times, label="naive")
        plt.legend()
        plt.show()

We can see that the performance of the fast methods is much better than
the performance of the naive algorithm. In order to see the differences
between the two fast methods, we will again compute them with more
samples. The large sample sizes used here could not be used with the naive
algorithm, as its used memory also grows quadratically.

.. jupyter-execute::
        
        n_samples_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        avl_times = np.zeros(len(n_samples_list))
        mergesort_times = np.zeros(len(n_samples_list))
        
        for i, n_samples in enumerate(n_samples_list):
        	x = np.random.normal(size=n_samples)
        	y = np.random.normal(size=n_samples)
        		
        	def avl():
        		return dcor.distance_covariance(x, y, method='AVL')
        		
        	def mergesort():
        		return dcor.distance_covariance(x, y, method='MERGESORT')
        		
        	avl_times[i] = timeit(avl, number=n_times)
        	mergesort_times[i] = timeit(mergesort, number=n_times)
        
        plt.title("Distance covariance performance comparison")
        plt.xlabel("Number of samples")
        plt.ylabel("Time (seconds)")
        plt.plot(n_samples_list, avl_times, label="avl")
        plt.plot(n_samples_list, mergesort_times, label="mergesort")
        plt.legend()
        plt.show()

References
----------
.. bibliography:: refs.bib
   :labelprefix: D
   :keyprefix: d-
