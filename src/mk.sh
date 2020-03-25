#!/bin/bash
# Some possible flags: ANIMATE=1 HOST_FLAGS="-march=native" PROFILE="cuda_profile"
GCCMAJORVER=`gcc -dumpversion | cut -d. -f1`
if [ "$GCCMAJORVER" != "4" ]
then
	make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 STD=14 $5
else
	make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 $5
fi
unset GCCMAJORVER
