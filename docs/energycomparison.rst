:html_theme.sidebar_secondary.remove:

Comparison with R's 'energy'
============================

The `'energy' <https://github.com/mariarizzo/energy>`_ package for R provides an implementation of the E-statistics in
this package programmed by the original authors of these statistics.

This package is inspired by 'energy', and tries to bring the same functionality
to a Python audience.

In this section, both packages are compared, to give an overview of the differences, and to make porting code
between R and Python easier.

Table of energy-dcor equivalents
--------------------------------

.. default-role:: py:obj

.. list-table::
    :widths: 25 25 100
    :header-rows: 1
    :class: comparison-table
    
    * - energy (R)
      - dcor (Python)
      - Notes
    * - .. code-block:: R
            
            dx <- dist(x)
            DX <- as.matrix(dx)
            
      - .. code-block:: python
            
            DX = dcor.distances.pairwise_distances(x)

      - Not really part of 'energy', but the 'stats' package.
      
        In Python it returns a numpy array, while in R it
        returns a matrix object
    * - .. code-block:: R
        
            D_center(DX)
            
      - .. code-block:: python
            
            dcor.double_centered(DX)
              
      -
    * - .. code-block:: R
        
            U_center(DX)
            
      - .. code-block:: python
            
            dcor.u_centered(DX)
              
      -   
    * - .. code-block:: R
        
             
            
      - .. code-block:: python
            
            dcor.mean_product(U, V)  
              
      - Provided for symmetry with `dcor.u_product`, although 
        the implementation is very simple
    * - .. code-block:: R
        
            U_product(U, V)
            
      - .. code-block:: python
            
            dcor.u_product(U, V) 
              
      -  
    * - .. code-block:: R
        
            
            
      - .. code-block:: python
            
            dcor.u_projection(U) 
              
      -
    * - .. code-block:: R
        
            
            
      - .. code-block:: python
            
            dcor.u_complementary_projection(U) 
              
      -  
    * - .. code-block:: R
        
            dcov(x, y)
            
      - .. code-block:: python
            
            dcor.distance_covariance(x, y) 
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            dcov(x, y, index = 0.5)
            
      - .. code-block:: python
            
            dcor.distance_covariance(
                x,
                y,
                exponent = 0.5,
            )
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            dcor(x, y)
            
      - .. code-block:: python
            
            dcor.distance_correlation(x, y) 
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            dcor(x, y, index = 0.5)
            
      - .. code-block:: python
            
            dcor.distance_correlation(
                x,
                y,
                exponent = 0.5,
            )
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            DCOR(x, y)
            
      - .. code-block:: python
            
            dcor.distance_stats(x, y) 
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            DCOR(x, y, index = 0.5)
            
      - .. code-block:: python
            
            dcor.distance_stats(
                x,
                y,
                exponent = 0.5,
            )
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            dcovU(x, y)
            
      - .. code-block:: python
            
            dcor.u_distance_covariance_sqr(x, y) 
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
             
            
      - .. code-block:: python
            
            dcor.u_distance_covariance_sqr(
                x,
                y,
                exponent = 0.5,
            )
              
      -
    * - .. code-block:: R
        
            bcdcor(x, y)
            
      - .. code-block:: python
            
            dcor.u_distance_correlation_sqr(x, y) 
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
             
            
      - .. code-block:: python
            
            dcor.u_distance_correlation_sqr(
                x,
                y,
                exponent = 0.5,
            )
              
      -
    * - .. code-block:: R
        
            dx <- dist(x)
            dy <- dist(y)
            
            DX <- as.matrix(dx)
            DY <- as.matrix(dy)
            
            dcovU_stats(DX, DY)
            
      - .. code-block:: python
            
            dcor.u_distance_stats_sqr(x, y) 
              
      - 
    * - .. code-block:: R
        
            
            
      - .. code-block:: python
            
            dcor.u_distance_stats_sqr(
                x,
                y,
                exponent = 0.5,
            ) 
              
      - 
    * - .. code-block:: R
        
            
            
      - .. code-block:: python
            
            dcor.distance_correlation_af_inv(x, y)
              
      - 
    * - .. code-block:: R
        
            pdcov(x, y, z)
            
      - .. code-block:: python
            
            dcor.partial_distance_covariance(x, y, z)
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            pdcor(x, y, z)
            
      - .. code-block:: python
            
            dcor.partial_distance_correlation(x, y, z)
              
      - In 'energy', the distance matrix can be computed
        beforehand. That is not currently possible in 'dcor'
    * - .. code-block:: R
        
            edist(
                rbind(x, y),
                c(nrow(x), nrow(y))
            )
            
      - .. code-block:: python
            
            dcor.homogeneity.energy_test_statistic(
                x,
                y,
            )
              
      - In spite of its name, 'energy' function 'edist' is not the energy distance,
        but a test statistic.
        
        The 'energy' version computes all pairwise estimations between clusters. The
        'dcor' version computes only the statistic between two random variables.
        
        The only method supported in 'dcor' is 'cluster'. 
    * - .. code-block:: R
        
            edist(
                rbind(x, y),
                c(nrow(x), nrow(y)), 
                alpha = 0.5,
                method="cluster"
            )
            
      - .. code-block:: python
            
            dcor.homogeneity.energy_test_statistic(
                x,
                y,
                exponent=0.5,
            )
              
      - In spite of its name, 'energy' function 'edist' is not the energy distance,
        but a test statistic.
        
        The 'energy' version computes all pairwise estimations between clusters. The
        'dcor' version computes only the statistic between two random variables.
        
        The only method supported in 'dcor' is 'cluster'.
    * - .. code-block:: R
        
            
            
      - .. code-block:: python
            
            dcor.energy_distance(x, y)
              
      - 
    * - .. code-block:: R
        
            eqdist.etest(
                rbind(x, y, z), 
                c(nrow(x), nrow(y), nrow(z)), 
                R=10
            )
            
      - .. code-block:: python
            
            dcor.homogeneity.energy_test(
                x,
                y,
                z, 
                num_resamples=10,
            )
              
      - Only the default method is implemented
    * - .. code-block:: R
        
            dcov.test(
                x,
                y,
                index = 0.5,
                R = 10
            )
            
      - .. code-block:: python
            
             dcor.independence.distance_covariance_test(
                 x,
                 y, 
                 exponent=0.5, 
                 num_resamples=10,
             )
              
      -
    * - .. code-block:: R
        
            dcor.t(x, y)
            
      - .. code-block:: python
            
             dcor.independence.distance_correlation_t_statistic(
                 x,
                 y,
             )
              
      -
    * - .. code-block:: R
        
            dcor.ttest(x, y)
            
      - .. code-block:: python
            
             dcor.independence.distance_correlation_t_test(
                 x,
                 y,
             )
              
      -
