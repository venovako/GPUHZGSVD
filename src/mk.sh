#!/bin/bash
# Some possible flags: ANIMATE=1 PROFILE="$2"
GCCMAJORVER=`gcc -dumpversion | cut -d. -f1`
if [ "$GCCMAJORVER" != "4" ]
then
	if [ "$GCCMAJORVER" != "9" ]
	then
		make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 STD=14 $5
	else
		make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 STD=17 $5
	fi
else
	make -C $1 GPU_ARCH="$2" NDEBUG=$3 CVG=$4 $5
fi
unset GCCMAJORVER
