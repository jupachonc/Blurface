#!/bin/sh
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 2"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
g++ -fopenmp videoFaceBlur.cpp -o videoFaceBlur `pkg-config --cflags --libs opencv`
echo "Compilación terminada, realizando pruebas ..."
echo "------------------------ Procesando videos------------"
echo "------------------------ Pruebas vídeo 1  ------------"
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 1 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 2 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 4 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 8 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 16 >> results.txt
echo "------------------------ Pruebas vídeo 2  ------------"
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 1 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 2 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 4 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 8 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 16 >> results.txt
echo "------------------------ Pruebas vídeo 3  ------------"
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 1 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 2 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 4 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 8 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 16 >> results.txt
echo "------------------------ Pruebas vídeo 4  ------------"
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 1 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 2 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 4 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 8 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 16 >> results.txt
echo "------------------------ Videos procesados consulte results.txt------------"
