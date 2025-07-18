#!/bin/bash
# Some possible flags: ANIMATE=1 PROFILE="$2"
make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 $5
