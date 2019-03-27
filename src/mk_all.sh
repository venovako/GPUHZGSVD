#!/bin/bash
for T in D Z
do
	for C in 0 1 2 3 4 5 6 7
	do
		./mk.sh $T $1 $2 $C $3
	done
done
