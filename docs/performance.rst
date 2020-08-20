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


Paralllel computation of distance covariance
--------------------------------------------

The following code explores some possible parallelizations of the distance
covariance. It used a fixed number of samples but modifies the number of
parallel computations.

.. jupyter-execute::
        
        from multiprocessing import Pool
        import dcor._fast_dcov_avl
        
        n_samples = 1000
        n_comps_list = [10, 50, 100]
        
        auto_map_times = np.zeros(len(n_samples_list))
        auto_parallel_times = np.zeros(len(n_samples_list))
        
        for i, n_comps in enumerate(n_comps_list):
        	x = np.random.normal(size=(n_comps, n_samples))
        	y = np.random.normal(size=(n_comps, n_samples))
        		
        	def auto_map():
        		return dcor.rowwise(dcor.distance_covariance, x, y)
        		
        	def vectorized():
        		return dcor.rowwise(dcor.distance_covariance, x, y,
        		                     map_function=Pool().map)
        		
        	def auto_parallel():
        		return dcor.rowwise(dcor.distance_covariance, x, y,
        		                     map_function=Pool().map)
        		
        	auto_map_times[i] = timeit(auto_map, number=n_times)
        	#auto_parallel_times[i] = timeit(auto_parallel, number=n_times)
        
        plt.title("Distance covariance performance comparison")
        plt.xlabel("Number of computations of distance covariance")
        plt.ylabel("Time (seconds)")
        plt.plot(n_samples_list, auto_map_times, label="map")
        plt.plot(n_samples_list, auto_parallel_times, label="map with pool of processes")
        plt.legend()
        plt.show()

References
----------
.. bibliography:: refs.bib
   :labelprefix: D
   :keyprefix: d-
