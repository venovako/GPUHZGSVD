# GPUHZGSVD
The Hari–Zimmermann generalized SVD for CUDA.

A part of the supplementary material for the paper
doi:[10.1177/1094342020972772](https://doi.org/10.1177/1094342020972772 "Implicit Hari–Zimmermann algorithm for the generalized SVD on the GPUs")
(arXiv:[1909.00101](https://arxiv.org/abs/1909.00101) \[math.NA\]).

## Building

### Prerequisites

A reasonably recent (e.g., 10.1.243) full CUDA installation on a 64-bit Linux (e.g., CentOS 7.7, optionally with devtoolset-8) is required.

For the Level 3 (multi-GPU) version an MPI installation on Linux built with the CUDA support (e.g., [Open MPI](https://www.open-mpi.org)) is required.

Then, clone and build [JACSD](https://github.com/venovako/JACSD) repository, with the same parent directory as this one.  In fact, only the ``jstrat`` library (i.e., ``libjstrat.a``) is needed to be built there.

### Make options

To build the test executable in ``double`` precision, do the following:
```bash
cd src
./mk.sh D SM OPT CVG
```
or, for ``double complex``,
```bash
cd src
./mk.sh Z SM OPT CVG
```
where ``SM`` is the target GPU architecture (e.g., for a Maxwell card it might be ``52``, for a Volta one ``70``, etc.), ``OPT`` is the optimization level (``3`` should be fine), and ``CVG`` is the algorithm requested (``0``, ``1``, ``2``, ``3``, ``4``, ``5``, ``6``, or ``7``).

It is also possible to append ``clean`` to the invocation above, to remove the executable, or such cleanup can be done manually.

For the Level 3 (multi-GPU) version, the ``prefix`` (e.g., ``/usr/local``) of your MPI distribution has to be provided:
```bash
cd src
./mk.sh Z SM OPT CVG MPI=prefix
```
Please, adjust the compiling and linking flags in the makefile(s) for your particular MPI distribution, since the flags provided therein have been tailored for Open MPI!

## Execution

### Command line

To run the executable, say, e.g.
```bash
/path/to/HZ0.exe DEV SNP0 SNP1 ALG MF MG N FN
```
where ``DEV`` is the CUDA device number, ``SNP0`` is the inner and ``SNP1`` outer strategy ID (``2`` for ``cycwor`` or ``4`` for ``mmstep``), ``ALG`` is ``0`` for full block or ``8`` for block-oriented, ``MF`` and ``MG`` are the number of rows of the first and the second matrix, respectively, ``N`` is the number of columns, and ``FN`` is the file name prefix (without an extension) containing the input data.

The Level 3 (multi-GPU) executables require a similar invocation:
```bash
/path/to/MHZ0.exe SNP0 SNP1 SNP2 ALG MF MG N FN
```
where ``SNP2`` is the outermost strategy ID (``3`` for ``cycwor`` or ``5`` for ``mmstep``; notice the increments), while the executable itself has to be run with at least two processes using ``mpiexec`` or a similar MPI job launcher.

### Data format

Data should be contained in ``FN.Y`` and ``FN.W`` binary, Fortran-array-order files, where the first one stores the matrix ``F`` and the second one the matrix ``G``, and both matrices are either ``double`` or ``double complex`` and are expected to have ``MF`` (first matrix) or ``MG`` (second matrix) rows and ``N`` columns.

The output comprises ``FN.YU``, ``FN.WV``, ``FN.Z``, for the ``double`` or ``double complex`` matrices ``U`` (``MF x N``), ``V`` (``MG x N``), and ``Z`` (``N x N``); and ``FN.SY``, ``FN.SW``, ``FN.SS``, for the ``double`` vectors ``\Sigma_F``, ``\Sigma_G``, and ``\Sigma``, respectively, where all vectors are of length ``N``.

See also [FLAPWxHZ](https://github.com/venovako/FLAPWxHZ) repository for more explanation.

This work has been supported in part by Croatian Science Foundation under the project IP-2014-09-3670 ([MFBDA](https://web.math.pmf.unizg.hr/mfbda/)).
