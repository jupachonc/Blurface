#!/bin/sh
echo "------------------------------------------------"
echo "Computación paralela y distribuida - práctica 1"
echo "------------------------------------------------"
echo "Compilando el programa ..."
#Compilar el programa
nvcc videoFaceBlur.cu -o videoFaceBlur -w `pkg-config --cflags --libs opencv`
echo "Compilación terminada, realizando pruebas ..."
echo "------------------------ Procesando videos------------"
echo "------------------------ Pruebas vídeo 1  ------------"
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 1 1 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 64 64 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 80 128 >> results.txt
./videoFaceBlur ./VideoIn/video_1.mp4 ./VideoOut/video_1_out.mp4 20 5 >> results.txt
echo "------------------------ Pruebas vídeo 2  ------------"
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 1 1 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 64 64 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 80 128 >> results.txt
./videoFaceBlur ./VideoIn/video_2.mp4 ./VideoOut/video_2_out.mp4 20 5 >> results.txt
echo "------------------------ Pruebas vídeo 3  ------------"
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 1 1 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 64 64 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 80 128 >> results.txt
./videoFaceBlur ./VideoIn/video_3.mp4 ./VideoOut/video_3_out.mp4 20 5 >> results.txt
echo "------------------------ Pruebas vídeo 4  ------------"
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 1 1 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 64 64 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 80 128 >> results.txt
./videoFaceBlur ./VideoIn/video_4.mp4 ./VideoOut/video_4_out.mp4 20 5 >> results.txt
echo "------------------------ Videos procesados consulte results.txt------------"
