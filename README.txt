README authors: Sorour E. Amiri, Liangzhe Chen and B. Aditya Prakash.
Dated: Dec 10, 2016

Note: You need to change the correct MATLAB_PATH in the makefile. You may need to include the directory of unordered_map and unordered_set in the makefile.

Usage:

First do 'make' (to compile sources)
then:
   python SnapNETS.py <data_path> <matlab_path> <num thread>

Example: python SnapNETS.py ./data/toy/ /Applications/MATLAB_R2014a.app/bin/matlab 2

==============================================================

To see a Demo: 
Again first do 'make'
then  
   'make demo'


==============================================================
Input:
graph.txt


infection.txt


nodes.txt

====================================================================
Output:
final_segmentation.txt

