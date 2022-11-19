#!/bin/sh
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 4"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
mpic++ -o videoFaceBlur videoFaceBlur.cpp -lm `pkg-config --cflags --libs opencv`
echo "Compilación terminada, realizando pruebas ..."
echo "------------------------ Procesando videos------------"
for ((v=1; v<=4; v+=1))
do
    printf "------------------------ Pruebas vídeo $v  ------------"
    for ((c=1; c<=4; c+=1))
    do
        mpirun -np $c --hostfile mpi_hosts ./videoFaceBlur ./VideoIn/video_"$v".mp4 ./VideoOut/video_"$v"_out.mp4 >> results.txt
    done
done
