#!/bin/bash
clang++ -framework OpenCL -stdlib=libc++ -std=gnu++11 gauss.cpp -o gauss.out
