#!/bin/bash
#PBS -A colosse-users
#PBS -l walltime=600
#PBS -l nodes=1:ppn=8

#PBS -o sortie.out
#PBS -e erreur.err

module load compilers/gcc
cd ~/GIF-7104-TPs/openMP/Jimmy

echo $(date +%c) 
./runBatch.sh
