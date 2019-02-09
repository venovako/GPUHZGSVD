# GPUHZGSVD
The Hari–Zimmermann generalized SVD for CUDA.

...work in progress...

## Building

### Prerequisites

A reasonably recent (e.g., 10.0) full CUDA installation on a 64-bit Linux or macOS is required.

Then, clone and build [GPUJACHx](https://github.com/venovako/GPUJACHx).
In fact only the ``strat.so`` library is needed.

### Make options

To build the test executable, do the following:
```bash
cd src
./mk.sh D SM OPT CVG
```
where ``SM`` is the target GPU architecture (e.g., for a Maxwell card it might be ``52``, for a Pascal one ``60``, etc.), ``OPT`` is the optimization level (``3`` should be fine), and ``CVG`` is the algorithm requested (``0``, ``1``, ``2``, ``3``, ``4``, ``5``, ``6``, or ``7``).

It is also possible to append ``clean`` to the invocation above, to remove the executable, or such cleanup can be done manually.

This work has been supported in part by Croatian Science Foundation under the project IP-2014-09-3670 ([MFBDA](https://web.math.pmf.unizg.hr/mfbda/)).
