.. _examples:

*****************************
Things you can do with Sphinx
*****************************

This documentation was created with a library called Sphinx.  With it you can easily create HTML without writing HTML.  Here is a list of examples you can do with Sphinx in order to create rich documentation simply for your software.

Latex
=======================================

Its easy to typeset equations:

.. math::

  S(f) = \sum_{n=0}^{N-1}s[n]e^{j2\pi f \frac{n}{N}}


Tables
======
+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | ...        | ...      |          |
+------------------------+------------+----------+----------+  

Emphasis
========

*emphasis*, **strong**, ``literal``

Lists
=====

* This is a bulleted list.
* It has two items, the second 
  	item uses two lines.

1.	This is a numbered list.
2.	It has items too.

#. 	This is also a numbered list.
#. 	It has two items too.

Nested lists

* this is
* a list

  * with a nested list
  * and some subitems

* and here the parent list continues

Figure
======

.. figure:: databaseFig.png
   :width: 400pt
   :alt: map to buried treasure

   This is the caption of the figure (a simple paragraph).

Footnotes
================   
Lorem ipsum [#f1]_ dolor sit amet ... [#f2]_

Embed Matplotlib figures
========================
.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   x = np.random.randn(1000)
   plt.hist( x, 20)
   plt.grid()
   plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
   plt.show()

.. rubric:: Footnotes

.. [#f1] Text of the first footnote.
.. [#f2] Text of the second footnote.