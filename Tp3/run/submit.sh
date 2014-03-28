#!/bin/bash

rm *.out *.err

msub runUser14.sh
showq -w user=user14
