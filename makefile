#Make sure you set a correct matlab path
CC = g++
#CFLAGS = -I /usr/include/c++/4.6/
CFLAGS = 
SOURCES=src/CoarseNet.cpp src/DisjointSets.cpp src/Edge.cpp src/graph.cpp
EXECUTABLE=src/CoarseNet

EXAMPLE1 = ./data/toy/
EXAMPLE2 = ./data/toy_dynamic/
THREAD1 = 1
THREAD2 = 1
 


$(EXECUTABLE): $(SOURCES)
	$(CC) $(SOURCES)  -std=c++0x $(CFLAGS) -o $(EXECUTABLE)

demo:
	python SnapNETS.py $(EXAMPLE1) $(THREAD1) $(THREAD2)

demo_dynamic:
	python SnapNETS_dynamic.py $(EXAMPLE2) $(THREAD1) $(THREAD2)

clean:
	rm -rf $(EXECUTABLE) 
