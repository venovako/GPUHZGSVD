#!/bin/bash
# Some possible flags: ANIMATE=1 HDF5_FULL="mpi,z" HOST_FLAGS="-ffp-contract=on,-integrated-as"
make GPU_ARCH=sm_$1 NDEBUG=$2 CVG=$3 $4
