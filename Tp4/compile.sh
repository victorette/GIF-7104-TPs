#!/bin/bash
clang++ -framework OpenCL -stdlib=libc++ -std=gnu++11 gauss.cpp -o invertMatrixOpenCL.out

##clang -framework OpenCL gauss.cpp -o invertMatrixOpenCL.out
