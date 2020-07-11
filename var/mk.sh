#!/bin/bash
GCCMAJORVER=`gcc -dumpversion | cut -d. -f1`
if [ "$GCCMAJORVER" != "4" ]
then
	if [ "$GCCMAJORVER" != "9" ]
	then
		make -C $1 GPU_ARCH="$2" NDEBUG=$3 STD=14 $4
	else
		make -C $1 GPU_ARCH="$2" NDEBUG=$3 STD=17 $4
	fi
else
	make -C $1 GPU_ARCH="$2" NDEBUG=$3 $4
fi
unset GCCMAJORVER
