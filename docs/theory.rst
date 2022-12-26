Theory
======

This section provides an explanation of the distance measures provided by this package
(distance covariance and distance correlation). The package can be used without a deep
understanding of the mathematics involved, so feel free to skip this chapter.

Distance covariance and distance correlation
--------------------------------------------

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
   \frac{\mathcal{V}^2(X, Y)}{\sqrt{\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y)}} &\text{ if
   $\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y) > 0$} \\
   0 &\text{ if $\mathcal{V}^2(X, X)\mathcal{V}^2(Y, Y) = 0$.}
   \end{cases}

Properties
^^^^^^^^^^

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
^^^^^^^^^^

Distance covariance has an estimator with a simple form. Suppose that we have
:math:`n` observations of :math:`X` and :math:`Y`, denoted by :math:`x` and :math:`y`. 
We denote as :math:`x_i` the 
:math:`i`-th observation of :math:`x`, and :math:`y_i` the :math:`i`-th observation of
:math:`y`. If we define :math:`a_{ij} = | x_i - x_j |_p` and :math:`b_{ij} = | y_i - y_j |_q`,
the corresponding double centered matrices (:func:`~dcor.double_centered`) are defined by :math:`(A_{i, j})_{i,j=1}^n`
and :math:`(B_{i, j})_{i,j=1}^n`

.. math::
   A_{i, j} &= a_{i,j} - \frac{1}{n} \sum_{l=1}^n a_{il} - \frac{1}{n}
   \sum_{k=1}^n a_{kj} + \frac{1}{n^2}\sum_{k,l=1}^n a_{kl}, \\
   B_{i, j} &= b_{i,j} - \frac{1}{n} \sum_{l=1}^n b_{il} - \frac{1}{n}
   \sum_{k=1}^n b_{kj} + \frac{1}{n^2}\sum_{k,l=1}^n b_{kl}.

Then

.. math::
   \mathcal{V}_n^2(x, y) = \frac{1}{n^2} \sum_{i,j=1}^n A_{i, j} B_{i, j}

is called the squared sample distance covariance (:func:`~dcor.distance_covariance_sqr`),
and it is an estimator of :math:`\mathcal{V}^2(X, Y)`. Its square root
(:func:`~dcor.distance_covariance`) is thus an estimator of the distance covariance.
The sample distance correlation
:math:`\mathcal{R}_n(x, y)` (:func:`~dcor.distance_correlation`) can be obtained as the
standardized sample covariance 

.. math::
   \mathcal{R}_n^2(x, y) = \begin{cases}
   \frac{\mathcal{V}_n^2(x, y)}{\sqrt{\mathcal{V}_n^2(x, x)\mathcal{V}_n^2(y, y)}},
   &\text{ if $\mathcal{V}_n^2(x, x)\mathcal{V}_n^2(y, y) > 0$}, \\
   0, &\text{ if $\mathcal{V}_n^2(x, x)\mathcal{V}_n^2(y, y) = 0$.}
   \end{cases}

These estimators have the following properties:

* :math:`\mathcal{V}_n^2(x, y) \geq 0`
* :math:`0 \leq \mathcal{R}_n^2(x, y) \leq 1`

In a similar way one can define an unbiased estimator :math:`\Omega_n(x, y)`
(:func:`~dcor.u_distance_covariance_sqr`) of the
squared distance covariance :math:`\mathcal{V}^2(X, Y)`. Given the
previous definitions of :math:`a_{ij}` and :math:`b_{ij}`, we define the :math:`U`-centered
matrices (:func:`~dcor.u_centered`) :math:`(\widetilde{A}_{i, j})_{i,j=1}^n` and :math:`(\widetilde{B}_{i, j})_{i,j=1}^n`

.. math::
   :label: ucentering
   
   \widetilde{A}_{i, j} &= \begin{cases} a_{i,j} - \frac{1}{n-2} \sum_{l=1}^n a_{il} -
   \frac{1}{n-2} \sum_{k=1}^n a_{kj} + \frac{1}{(n-1)(n-2)}\sum_{k,l=1}^n a_{kl}, &\text{if } i \neq j, \\
   0, &\text{if } i = j,
   \end{cases} \\
   \widetilde{B}_{i, j} &= \begin{cases} b_{i,j} - \frac{1}{n-2} \sum_{l=1}^n b_{il} -
   \frac{1}{n-2} \sum_{k=1}^n b_{kj} + \frac{1}{(n-1)(n-2)}\sum_{k,l=1}^n b_{kl}, &\text{if } i \neq j, \\
   0, &\text{if } i = j.
   \end{cases}

Then, :math:`\Omega_n(x, y)` is defined as

.. math::
   \Omega_n(x, y) = \frac{1}{n(n-3)} \sum_{i,j=1}^n \widetilde{A}_{i, j}
   \widetilde{B}_{i, j}.

We can also obtain an estimator of :math:`\mathcal{R}^2(X, Y)`
(:func:`~dcor.u_distance_correlation_sqr`) using :math:`\Omega_n(x, y)`,
as we did with :math:`\mathcal{V}_n^2(x, y)`. :math:`\Omega_n(x, y)` does not verify that
:math:`\Omega_n(x, y) \geq 0`, because sometimes could take negative values near :math:`0`.

There are algorithms that can compute :math:`\mathcal{V}_n^2(x, y)` and :math:`\Omega_n(x, y)`
for random variables with :math:`O(n\log n)` complexity
:cite:`c-fast_distance_correlation_avl,c-fast_distance_correlation_mergesort`. Since
the estimator formulas explained above have complexity :math:`O(n^2)`, this
improvement is significant, specially for larger samples.

Partial distance covariance and partial distance correlation
------------------------------------------------------------

Partial distance covariance and partial distance correlation are dependency measures
between random vectors, based on distance covariance and distance correlation, in with
the effect of a random vector is removed :cite:`c-partial_distance_correlation`. 
The population partial distance covariance :math:`\mathcal{V}^{*}(X, Y; Z)`, or
:math:`\text{pdCov}^{*}(X, Y; Z)`, between two random vectors :math:`X` and 
:math:`Y` with respect to a random vector :math:`Z` is

.. math::
   \mathcal{V}^{*}(X, Y; Z) = \begin{cases}
   \mathcal{V}^2(X, Y) - 
   \frac{\mathcal{V}^2(X, Z)\mathcal{V}^2(Y, Z)}{\mathcal{V}^2(Z, Z)} & \text{if } 
   \mathcal{V}^2(Z, Z) \neq 0 \\
   \mathcal{V}^2(X, Y) & \text{if } 
   \mathcal{V}^2(Z, Z) = 0
   \end{cases}
   
where :math:`\mathcal{V}^2({}\cdot{}, {}\cdot{})` is the squared distance covariance.
   
The corresponding partial distance correlation :math:`\mathcal{R}^{*}(X, Y; Z)`, or
:math:`\text{pdCor}^{*}(X, Y; Z)`, is

.. math::
   \mathcal{R}^{*}(X, Y; Z) = \begin{cases}
   \frac{\mathcal{R}^2(X, Y) - 
   \mathcal{R}^2(X, Z)\mathcal{R}^2(Y, Z)}{\sqrt{1 - \mathcal{R}^4(X, Z)}\sqrt{1 - \mathcal{R}^4(Y, Z)}} 
   & \text{if } \mathcal{R}^4(X, Z) \neq 1 \text{ and } \mathcal{R}^4(Y, Z) \neq 1 \\
   0
   & \text{if } \mathcal{R}^4(X, Z) = 1 \text{ or } \mathcal{R}^4(Y, Z) = 1
   \end{cases}
   
where :math:`\mathcal{R}({}\cdot{}, {}\cdot{})` is the distance correlation.

Estimators
^^^^^^^^^^

As in distance covariance and distance correlation, the :math:`U`-centered
distance matrices :math:`\widetilde{A}_{i, j}`, :math:`\widetilde{B}_{i, j}` and 
:math:`\widetilde{C}_{i, j}` corresponding with the samples :math:`x`, :math:`y` and
:math:`z` taken from the random vectors :math:`X`, :math:`Y` and
:math:`Z` can be computed using using :eq:`ucentering`.

The set of all :math:`U`-centered distance matrices is a Hilbert space with the inner product (:func:`~dcor.u_product`)

.. math::
   \langle \widetilde{A}, \widetilde{B} \rangle = \frac{1}{n(n-3)} \sum_{i,j=1}^n 
   \widetilde{A}_{i, j} \widetilde{B}_{i, j}.
   
Then, the projection of a sample :math:`x` over :math:`z` (:func:`~dcor.u_projection`) can be taken
in this Hilbert space using the associated matrices, as

.. math::
   P_z(x) = \frac{\langle \widetilde{A}, \widetilde{C} \rangle}{\langle \widetilde{C}, 
   \widetilde{C} \rangle}\widetilde{C}.
   
The complementary projection (:func:`~dcor.u_complementary_projection`) is then

.. math::
   P_{z^{\perp}}(x) = \widetilde{A} - P_z(x) = \widetilde{A} - \frac{\langle \widetilde{A},
   \widetilde{C} \rangle}{\langle \widetilde{C}, \widetilde{C} \rangle}\widetilde{C}.
   
We can now define the sample partial distance covariance
(:func:`~dcor.partial_distance_covariance`) as

.. math::
   \mathcal{V}_n^{*}(x, y; z) = \langle P_{z^{\perp}}(x), P_{z^{\perp}}(y) \rangle
   
The sample distance correlation (:func:`~dcor.partial_distance_correlation`) is defined as
the cosine of the angle between the vectors :math:`P_{z^{\perp}}(x)` and :math:`P_{z^{\perp}}(y)`

.. math::
   \mathcal{R}_n^{*}(x, y; z) = \begin{cases} 
   \frac{\langle P_{z^{\perp}}(x), P_{z^{\perp}}(y) \rangle}{||P_{z^{\perp}}(x)||
   ||P_{z^{\perp}}(y)||} & \text{if } ||P_{z^{\perp}}(x)|| ||P_{z^{\perp}}(y)|| \neq 0 \\
   0 & \text{if } ||P_{z^{\perp}}(x)|| ||P_{z^{\perp}}(y)|| = 0 
   \end{cases} 

Energy distance
---------------

Energy distance is an statistical distance between random vectors :math:`X, Y \in \mathbb{R}^d` 
:cite:`c-energy_distance`, defined as

.. math::
   \mathcal{E}(X, Y) = 2\mathbb{E}(|| X - Y ||) - \mathbb{E}(|| X - X' ||) - 
   \mathbb{E}(|| Y - Y' ||)

where :math:`X'` and :math:`Y'` are independent and identically distributed copies of
:math:`X` and :math:`Y`, respectively.

It can be proved that, if the characteristic functions of :math:`X` and :math:`Y` are
:math:`\phi_X(t)` and :math:`\phi_Y(t)` the energy distance can be alternatively written
as

.. math::
   \mathcal{E}(X, Y) = \frac{1}{c_d} \int_{\mathbb{R}^d}
   \frac{|\phi_X(t) - \phi_Y(t)|^2}{||t||^{d+1}}dt

where again :math:`c_d = \frac{\pi^{(1 + d)/2}}{\Gamma((1 +
d)/2)}` is half the surface area of the unit sphere in :math:`\mathbb{R}^d`.

Estimator
^^^^^^^^^

Suppose that we have :math:`n_1` observations of :math:`X` and :math:`n_2` observations of 
:math:`Y`, denoted by :math:`x` and :math:`y`. We denote as :math:`x_i` the 
:math:`i`-th observation of :math:`x`, and :math:`y_i` the :math:`i`-th observation of
:math:`y`. Then, an estimator of the energy distance (:func:`~dcor.energy_distance`) is

.. math::
   \mathcal{E_{n_1, n_2}}(x, y) = \frac{2}{n_1 n_2}\sum_{i=1}^{n_1}\sum_{j=1}^{n_2}|| x_i - y_j ||
   - \frac{1}{n_1^2}\sum_{i=1}^{n_1}\sum_{j=1}^{n_1}|| x_i - x_j ||
   - \frac{1}{n_2^2}\sum_{i=1}^{n_2}\sum_{j=1}^{n_2}|| y_i - y_j ||

References
----------
.. bibliography:: refs.bib
   :labelprefix: C
   :keyprefix: c-
