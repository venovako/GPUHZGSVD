#!/bin/bash
# PLEASE ADAPT TO YOUR OWN BUILD SYSTEM
echo "Building test_dsygvd.exe ..."
scl enable gcc-toolset-14 "nvcc -DNDEBUG -DUSE_CUSOLVER -Ishared -std c++20 -restrict -O3 --extra-device-vectorization -res-usage -Xcompiler -march=native -Xcompiler -pthread -gencode arch=compute_$1,code=sm_$1 test_dsygvd.cu shared/cuda_helper.cu shared/my_utils.cu -o test_dsygvd.exe -l cusolver,cublas,dl,m"
echo "done."

echo "Building test_zhegvd.exe ..."
scl enable gcc-toolset-14 "nvcc -DNDEBUG -DUSE_CUSOLVER -Ishared -std c++20 -restrict -O3 --extra-device-vectorization -res-usage -Xcompiler -march=native -Xcompiler -pthread -gencode arch=compute_$1,code=sm_$1 test_zhegvd.cu shared/cuda_helper.cu shared/my_utils.cu -o test_zhegvd.exe -l cusolver,cublas,dl,m"
echo "done."

echo "Building test_compare.exe ..."
scl enable gcc-toolset-14 "gcc -DNDEBUG -Ishared -O3 -march=native -pthread test_compare.c -o test_compare.exe -lm"
echo "done."
