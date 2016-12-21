#Make sure you set a correct matlab path
CC = g++
#CFLAGS = -I /usr/include/c++/4.6/
CFLAGS = 
SOURCES=src/CoarseNet.cpp src/DisjointSets.cpp src/Edge.cpp src/graph.cpp
EXECUTABLE=src/CoarseNet

EXAMPLE = ./data/toy/
THREAD = 1
MATLAB_PATH = /Applications/MATLAB_R2014a.app/bin/matlab  


$(EXECUTABLE): $(SOURCES)
	$(CC) $(SOURCES)  -std=c++0x $(CFLAGS) -o $(EXECUTABLE)

demo:
	python SnapNETS.py $(EXAMPLE) $(MATLAB_PATH) $(THREAD)

clean:
	rm -rf $(EXECUTABLE) 
