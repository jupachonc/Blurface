CC =    g++ -fopenmp
PROJECT =   videoFaceBlur
SRC =   videoFaceBlur.cpp
LIBS =  `pkg-config --cflags --libs opencv4`
$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS)