#!/bin/sh
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 1"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
g++ videoFaceBlur.cpp -o videoFaceBlur `pkg-config --cflags --libs opencv4`
echo "Compilación terminada, realizando pruebas ..."
echo "------------------------ Procesando videos------------"
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 >> results.txt
echo "------------------------ Videos procesados consulte results.txt------------"
