#!/bin/bash
#PBS -A colosse-users
#PBS -l walltime=600
#PBS -l nodes=4:ppn=8
#PBS -r n

#PBS -o sortie_4.out
#PBS -e erreur_4.err

module load compilers/gcc/4.8.0
module load mpi/openmpi/1.6.4_gcc
module load apps/blcr/0.8.4

cd ~/GIF-7104-TPs/Tp3

echo $(date +%c) 
./runBatch.sh
