# GPUHZGSVD
The Hari–Zimmermann generalized SVD for CUDA.

...work in progress...

## Building

### Prerequisites

A reasonably recent (e.g., 9.2) full CUDA installation on a 64-bit Linux or macOS is required.

Have the HDF5 library (including the high-level interface) installed (e.g., in ``$HOME\hdf5``).

Then, clone and build [GPUJACHx](https://github.com/venovako/GPUJACHx).
In fact only the ``strat.so`` library is needed.

### Make options

To build the test executable, do the following:
```bash
cd src
./mk.sh SM OPT CVG
```
where ``SM`` is the target GPU architecture (e.g., for a Maxwell card it might be ``52``, for a Pascal one ``60``, etc.), ``OPT`` is the optimization level (``3`` should be fine), and ``CVG`` is the algorithm requested (``0`` and ``1`` done, ``2`` and ``3`` in progress).

It is also possible to append ``clean`` to the invocation above, to remove the executable, or such cleanup can be done manually.
