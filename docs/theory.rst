Theory
======

This section provides an explanation of the distance measures provided by this package
(distance covariance and distance correlation). The package can be used without a deep
understanding of the mathematics involved, so feel free to skip this chapter.

Definition of distance covariance and distance correlation
----------------------------------------------------------

Distance covariance and distance correlation are recently introduced dependency
measures between random vectors :cite:`c-distance_correlation`. Let :math:`X` and :math:`Y` be
two random vectors with finite first moments, and let :math:`\phi_X` and :math:`\phi_Y` be
the respective characteristic functions

.. math::
   \phi_X(t) &= \mathbb{E}[e^{itX}] \\
   \phi_Y(t) &= \mathbb{E}[e^{itY}]
   
Let :math:`\phi_{X, Y}` be the joint
characteristic function. Then, if :math:`X` and :math:`Y` take values in :math:`\mathbb{R}^p` and 
:math:`\mathbb{R}^q` respectively, the distance covariance between them :math:`\mathcal{V}(X, Y)`, or
:math:`\text{dCov}(X, Y)`, is the non-negative number defined by

.. math::
   \mathcal{V}^2(X, Y) = \int_{\mathbb{R}^{p+q}}|\phi_{X, Y}(t, s) -
   \phi_X(t)\phi_Y(s)|^2w(t,s)dt ds,
   
where :math:`w(t, s) = (c_p c_q |t|_p^{1+p}|s|_q^{1+q})^{-1}`, :math:`|{}\cdot{}|_d` is
the euclidean norm in :math:`\mathbb{R}^d` and :math:`c_d = \frac{\pi^{(1 + d)/2}}{\Gamma((1 +
d)/2)}` is half the surface area of the unit sphere in :math:`\mathbb{R}^d`. The distance
correlation :math:`\mathcal{R}(X, Y)`, or :math:`\text{dCor}(X, Y)`, is defined as

.. math::
   \mathcal{R}^2(X, Y) = \begin{cases}
   \frac{\mathcal{V}^2(X, Y)}{\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y)} &\text{ if
   $\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y) > 0$} \\
   0 &\text{ if $\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y) = 0$.}
   \end{cases}

Properties
----------

The distance covariance has the following properties:

* :math:`\mathcal{V}(X, Y) \geq 0`.
* :math:`\mathcal{V}(X, Y) = 0` if and only if :math:`X` and :math:`Y` are independent.
* :math:`\mathcal{V}(X, Y) = \mathcal{V}(Y, X)`.
* :math:`\mathcal{V}^2(\mathbf{a}_1 + b_1 \mathbf{C}_1 X, \mathbf{a}_2 + b_2
  \mathbf{C}_2 Y) = |b_1 b_2| \mathcal{V}^2(Y, X)` for all constant
  real-valued vectors :math:`\mathbf{a}_1, \mathbf{a}_2`, scalars :math:`b_1, b_2` and
  orthonormal matrices :math:`\mathbf{C}_1, \mathbf{C}_2`.
* If the random vectors :math:`(X_1, Y_1)` and :math:`(X_2, Y_2)` are independent then
  
.. math::
   \mathcal{V}(X_1 + X_2, Y_1 + Y_2) \leq \mathcal{V}(X_1, Y_1) +
   \mathcal{V}(X_2, Y_2).

The distance correlation has the following properties:

  * :math:`0 \leq \mathcal{R}(X, Y) \leq 1`.
  * :math:`\mathcal{R}(X, Y) = 0` if and only if :math:`X` and :math:`Y` are independent.
  * If :math:`\mathcal{R}(X, Y) = 1` then there exists a vector :math:`\mathbf{a}`, a
    nonzero real number :math:`b` and an orthogonal matrix :math:`\mathbf{C}` such that :math:`Y =
    \mathbf{a} + b\mathbf{C}X`.
  
Estimators
----------

Distance covariance has an estimator with a simple form. Suppose that we have
:math:`n` observations of :math:`X` and :math:`Y`. We denote as :math:`X_i` the 
:math:`i`-th observation of :math:`X`, and :math:`Y_i` the :math:`i`-th observation of
:math:`Y`. If we define :math:`a_{ij} = | X_i - X_j |_p` and :math:`b_{ij} = | Y_i - Y_j |_q`,
the corresponding double centered matrices are defined by :math:`(A_{i, j})_{i,j=1}^n`
and :math:`(B_{i, j})_{i,j=1}^n`

.. math::
   A_{i, j} &= a_{i,j} - \frac{1}{n} \sum_{l=1}^n a_{il} - \frac{1}{n}
   \sum_{k=1}^n a_{kj} + \frac{1}{n^2}\sum_{k=1}^n a_{kj}, \\
   B_{i, j} &= b_{i,j} - \frac{1}{n} \sum_{l=1}^n b_{il} - \frac{1}{n}
   \sum_{k=1}^n b_{kj} + \frac{1}{n^2}\sum_{k=1}^n b_{kj}.

Then

.. math::
   \mathcal{V}_n^2(X, Y) = \frac{1}{n^2} \sum_{i,j=1}^n A_{i, j} B_{i, j}

is called the squared sample distance covariance, and it is an estimator of
:math:`\mathcal{V}^2(X, Y)`. The sample distance correlation :math:`\mathcal{R}_n(X, Y)`
can be obtained as the standardized sample covariance 

.. math::
   \mathcal{R}_n^2(X, Y) = \begin{cases}
   \frac{\mathcal{V}_n^2(X, Y)}{\mathcal{V}_n^2(X, X)\mathcal{V}_n^2(Y, Y)},
   &\text{ if $\mathcal{V}_n^2(X, X)\mathcal{V}_n^2(Y, Y) > 0$}, \\
   0, &\text{ if $\mathcal{V}_n^2(X, X)\mathcal{V}_n^2(Y, Y) = 0$.}
   \end{cases}

These estimators have the following properties:

* :math:`\mathcal{V}_n^2(X, Y) \geq 0`
* :math:`0 \leq \mathcal{R}_n^2(X, Y) \leq 1`

In a similar way one can define an unbiased estimator :math:`\Omega_n(X, Y)` of the
squared distance covariance :math:`\mathcal{V}^2(X, Y)`. Given the
previous definitions of :math:`a_{ij}` and :math:`b_{ij}`, we define the :math:`U`-centered
matrices :math:`(\widetilde{A}_{i, j})_{i,j=1}^n` and :math:`(\widetilde{B}_{i, j})_{i,j=1}^n`

.. math::
   \widetilde{A}_{i, j} &= \begin{cases} a_{i,j} - \frac{1}{n-2} \sum_{l=1}^n a_{il} -
   \frac{1}{n-2} \sum_{k=1}^n a_{kj} + \frac{1}{(n-1)(n-2)}\sum_{k=1}^n a_{kj}, &\text{if } i \neq j, \\
   0, &\text{if } i = j,
   \end{cases} \\
   \widetilde{B}_{i, j} &= \begin{cases} b_{i,j} - \frac{1}{n-2} \sum_{l=1}^n b_{il} -
   \frac{1}{n-2} \sum_{k=1}^n b_{kj} + \frac{1}{(n-1)(n-2)}\sum_{k=1}^n b_{kj}, &\text{if } i \neq j, \\
   0, &\text{if } i = j.
   \end{cases}

Then, :math:`\Omega_n(X, Y)` is defined as

.. math::
   \Omega_n(X, Y) = \frac{1}{n(n-3)} \sum_{i,j=1}^n \widetilde{A}_{i, j}
   \widetilde{B}_{i, j}.

We can also obtain an estimator of :math:`\mathcal{R}^2(X, Y)` using :math:`\Omega_n(X, Y)`,
as we did with :math:`\mathcal{V}_n^2(X, Y)`. :math:`\Omega_n(X, Y)` does not verify that
:math:`\Omega_n(X, Y) \geq 0`, because sometimes could take negative values near :math:`0`.
There is an algorithm that can compute :math:`\Omega_n(X, Y)` for random variables
with :math:`O(n\log n)` complexity :cite:`c-fast_distance_correlation`. Since
the estimator formulas explained above have complexity :math:`O(n^2)`, this
improvement is significant, specially for larger samples.

References
----------
.. bibliography:: refs.bib
   :labelprefix: C
   :keyprefix: c-
