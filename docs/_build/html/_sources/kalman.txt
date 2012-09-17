.. _kalman:

*****************************
Kalman Filter
*****************************

Definitions
===========

| :math:`P_k`:    Prior Error Covariance
| :math:`\hat{x}_k`:    Current State Estimate
| :math:`\hat{x}'_k`:    Current State Estimate before update step
| :math:`H`:    Measurement Matrix
| :math:`A`:    State Matrix
| :math:`K`: Kalman Gain

Prediction
==========
Predict State
-------------
.. math::
  
  \hat{x}'_{k} = A\hat{x}_{k-1}

.. sourcecode:: python

  from numpy import dot, inv

  X = dot(A, X)

Predict Covariance
------------------
.. math::
  
  P_{k} = AP_{k-1}A^T + Q

.. sourcecode:: python

  from numpy import dot, inv

  P = dot(A, dot(P, A.T)) + Q

Update
============
Kalman Gain
-----------
.. math::
  
  K_k = P'_kH^T(HP'_kH^T + R)^{-1}

.. sourcecode:: python

  from numpy import dot, inv

  S = dot(H, dot(P, H.T)) + R
  K = dot(P, dot(H.T, inv(S))) 

Update Estimate
-----------
.. math::
  
  \hat{x}_k = \hat{x}'_k + K_k(y_k - H\hat{x}'_k)

.. sourcecode:: python

  from numpy import dot

  M = dot(H, X)  
  X = X + dot(K, (Y - M))

Update Covariance
-----------
.. math::
  
  P_k = P'_k - K_kHP'_k

.. sourcecode:: python

  from numpy import dot

  P = P - dot(K, dot(H, P))

