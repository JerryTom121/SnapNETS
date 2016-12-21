Authors: Sorour E. Amiri, Liangzhe Chen, and B. Aditya Prakash
Date: Dec 10, 2016

Note: You need to change the correct MATLAB_PATH in the makefile.

Usage:

>> make
>> make demo

First do 'make' (to compile sources). Then 'make demo' will run the Snapnets for toy example. You can directly run SnapNETS.py with the following command:

>> python SnapNETS.py <data_path> <matlab_path> <num thread>

<data_path> : Directory of the dataset
<matlab_path>: Matlab directory in your machine
<num thread>: Number of processors for parallel implementation

Example: python SnapNETS.py ./data/toy/ /Applications/MATLAB_R2014a.app/bin/matlab 2

==============================================================
Input:
- graph.txt : It is a tab separated file and index of nodes starts from 1 and are consecutive. Here is an example graph and its representation:
1 ----- 2
 |	    |
 |      |
 |	    |
 3 ----- 5
  \     /
   \   /
    \ /
     4

The graph.txt file is:

Source	Target
1	2
1	3
3	4
3	5
5	4


- infection.txt: It is a tab separated file. It shows when a node gets activated. If a node remains deactive in the entire sequence, it will not node appear in the file. Here is an example of infection.txt:

node	time
1	1.1
2	1.2
3	1.3



====================================================================
Output:
- final_segmentation.txt:  It shows the final segmentation result. For example,
'1.1-2.1','2.1-4.1',

It means we have a cut point at time 2.1 in the time interval 1.1-4.1

- Intermediate results:
     * 90: It is a directory which contains the coarsened graphs and their feature representations. The following are the intermediate files correspond to the first snapshot of the above example:
         coarse_0_seg_1.1_2.1
         feature0.txt
         final_map_0_seg_1.1_2.1.txt
         graph0_1.1_2.1.txt
         nodes0_1.1_2.1.txt
         time_0_seg_1.1_2.1.txt

    * cc: It is a directory which contains the active nodes in each snapshot and the score of edges to merge in the coarsening process. 
