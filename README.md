# Intention of the Source Files

The files were used in order to create the results for the paper [Bivariate Estimation-of-Distribution Algorithms Can Find an Exponential Number of Optima](https://dl.acm.org/doi/10.1145/3377930.3390177).
They act as a means to clarify implementation details of the algorithm, the objective function, as well as the test setup.
However, the files are not necessarily intended to be used out of the box, although all files with a ``main`` can be ran (that is, all files but ``fitness_functions.d``, ``mimic.d``, and ``mimic_looking_for_more_optima.d``), resulting in data that was used in the paper.

The code is written in D, for which [several compilers exist](https://dlang.org/download.html).
Any compiler that works for your system is well suited.

**Important:** Running experiments that save all of the different optima found during the optimization process (``number_of_different_optima.d``) results in *massive* files (about 90 GB!), depending on the dimension size of the problem.
