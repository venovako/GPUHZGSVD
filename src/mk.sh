#!/bin/bash
# Some possible flags: STD=14 ANIMATE=1 HOST_FLAGS="-ffp-contract=on,-integrated-as"
make -C $1 GPU_ARCH=$2 NDEBUG=$3 CVG=$4 $5
