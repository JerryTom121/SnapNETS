#Make sure you set a correct matlab path
CC = g++
#CFLAGS = -I /usr/include/c++/4.6/
CFLAGS = 
SOURCES=src/CoarseNet.cpp src/DisjointSets.cpp src/Edge.cpp src/graph.cpp
EXECUTABLE=src/CoarseNet

EXAMPLE = example/oregon.inf
PERCENT = 50
MATLAB_PATH = /usr/local/R2011B/bin/matlab  #/Applications/MATLAB_R2014a.app/bin/matlab


$(EXECUTABLE): $(SOURCES)
	$(CC) $(SOURCES)  -std=c++0x $(CFLAGS) -o $(EXECUTABLE)

demo:
	python coarse_net.py $(EXAMPLE) $(PERCENT) $(MATLAB_PATH)

clean:
	rm -rf $(EXECUTABLE) ./example/*_coarse_* ./example/*_final_map_* ./example/*_scores ./example/*_time 
