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

To build the test executable ``HZ1.exe``, execute:
```bash
cd src
./mk.sh SM OPT
```
where ``SM`` is the target GPU architecture (e.g., for a Maxwell card it might be ``52``, for a Pascal one ``60``, etc.), and ``OPT`` is the optimization level (``3`` should be fine).

It is also possible to append ``clean`` to the invocation above, to remove ``HZ1.exe``, or such cleanup can be done manually.
