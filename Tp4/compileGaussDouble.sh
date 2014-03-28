#!/bin/bash
clang++ -framework OpenCL -stdlib=libc++ -std=gnu++11 gaussDouble.cpp -o gaussDouble.out

##clang -framework OpenCL gauss.cpp -o invertMatrixOpenCL.out
