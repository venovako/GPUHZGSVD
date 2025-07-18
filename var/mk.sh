#!/bin/bash
make -C $1 GPU_ARCH="$2" NDEBUG=$3 $4
