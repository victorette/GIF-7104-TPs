#!/bin/bash

echo $(date +%c)

echo -e "\n"

echo -e "Version fourni par l\'enseignant sans arguments.\n"

./tp2Original

echo "\nVersion sequentielle optimise."

#./tp2Sequential 1000000000

echo -e "\nVersion multifilaire avec un seul fil d'execution."
./tp2OpenMp 1000000000 1

echo -e "\nVersion multifilaire avec deux fil d'execution."
./tp2OpenMp 1000000000 2

echo -e "\nVersion multifilaire avec trois fil d'execution."
./tp2OpenMp 1000000000 3

echo -e "\nVersion multifilaire avec quatre fil d'execution."
./tp2OpenMp 1000000000 4

echo -e "\nVersion multifilaire avec cinq fil d'execution."
./tp2OpenMp 1000000000 5

echo -e "\nVersion multifilaire avec six fil d'execution."
./tp2OpenMp 1000000000 6

echo -e "\nVersion multifilaire avec sept fil d'execution."
./tp2OpenMp 1000000000 7

echo -e "\nVersion multifilaire avec huit fil d'execution."
./tp2OpenMp 1000000000 8

