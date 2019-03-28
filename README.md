# GPUHZGSVD
The Hari–Zimmermann generalized SVD for CUDA.

## Building

### Prerequisites

A reasonably recent (e.g., 10.1) full CUDA installation on a 64-bit Linux or macOS is required.

Then, clone and build [GPUJACHx](https://github.com/venovako/GPUJACHx).
In fact only the ``strat.so`` library is needed to be built there.

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

## Execution

### Command line

To run the executable, say, e.g.
```bash
/path/to/HZ0.exe DEV SDY SNP0 SNP1 ALG M N FN
```
where ``DEV`` is the CUDA device number, ``SDY`` is a path to ``strat.so``, ``SNP0`` is the inner and ``SNP1`` outer strategy name (``cycwor`` or ``mmstep``), ``ALG`` is ``0`` for full block or ``8`` for block-oriented, ``M`` and ``N`` are the number of rows and columns, respectively, and ``FN`` is the file name prefix (without an extension) containing the input data.

### Data format

Data should be contained in ``FN.Y`` and ``FN.W`` binary, Fortran-array-order files, where the first one stores the matrix ``F`` and the second one the matrix ``G``, and both matrices are either ``double`` or ``double complex`` and are expected to have ``M`` rows and ``N`` columns.

The output comprises ``FN.YU``, ``FN.WV``, ``FN.Z``, for the ``double`` or ``double complex`` matrices ``U``, ``V`` (both ``M x N``), and ``Z`` (``N x N``); and ``FN.SY``, ``FN.SW``, ``FN.SS``, for the ``double`` vectors ``\Sigma_F``, ``\Sigma_G``, and ``\Sigma``, respectively, where all vectors are of length ``N``.

Please, consult also [FLAPWxHZ](https://github.com/venovako/FLAPWxHZ) repository.

This work has been supported in part by Croatian Science Foundation under the project IP-2014-09-3670 ([MFBDA](https://web.math.pmf.unizg.hr/mfbda/)).
