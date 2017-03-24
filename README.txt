Authors: Sorour E. Amiri, Liangzhe Chen, and B. Aditya Prakash
Paper Title: SnapNETS: Automatic Segmentation of Network Sequences with Node Labels
Date: March 2, 2017

Usage:
To run SnapNets on static toy graph with fixed structure do as follows,
>> make
>> make demo  

To run SnapNets on temporal toy graph with dynamic structure do as follows, 
>> make
>> make demo_daynamic



First do 'make' (to compile sources). Then 'make demo'/'make demo_daynamic' will run the Snapnets/SnapNETS_dynamic for toy example. You can directly run SnapNETS with the following command:
 
To run SnapNets on static graphs with fixed structure:
>> python SnapNETS.py <data_path> <num thread 1> <num thread 2>

To run SnapNets on temporal graphs with dynamic structure do as follows, 
>> python SnapNETS_dynamic.py <data_path> <num thread 1> <num thread 2>

<data_path> : Directory of the dataset
<num thread 1>: Number of processors to summaries graphs in parallel
<num thread 2>: Number of processors to generate the segmentation graphs in parallel

Example: 
    Static graphs:   python SnapNETS.py ./data/toy/ 1 1
    Dynamic graphs:  python SnapNETS_dynamic.py ./data/toy_dynamic/ 1 1



==============================================================
Input:

- links.txt :

It is a tab separated file and index of nodes starts from 1 and are consecutive. Here are an example graph and its representation:
 1 ----- 2
 |       |
 |       |
 |       |
 3 ----- 5
  \     /
   \   /
    \ /
     4

The links.txt file for SnapNETS.py looks like as follows:

Source	Target
1	2
1	3
3	4
3	5
5	4

The links.txt file for SnapNETS_dynamic.py looks like as follows:

Source  Target  appearance_time  removal_time
1   2   1   5
1   3   1   -
3   4   -   4
3   5   -   -
5   4   -   -

For example "1   2   1   5" means there is a link between node 1 and node 2 and in appeared in the first snapshot and it removed in the fifth snapshot. "3   5   -   -" means the link between 3 and 5 is in the graph from the beginning to the end of the sequence.

- actives.txt: 
In both SnapNETS.py and SnapNETS_dynamic.py, the format of active.txt is the same.
It is a tab separated file. It shows when a node is active. If a node remains deactive in the entire sequence, it will not appear in the file. Here is an example of actives.txt:

node	times
1   1
2   2   3
3   2   3
4   2   3

It means node 1 is active on the first timestamp and node 2 is active on the second and third timestamp.

====================================================================
Output:
- final_segmentation.txt:  It shows the final segmentation result. For example,
1.0- 3.0    3.0- 5.0

It means SnapNETs detects a cut point at time 3.0 in the time interval 1.0-5.0

- Intermediate results:
     * 90: It is a directory which contains the coarsened graphs and their feature representations. The following are the intermediate files correspond to the first snapshot of the above example:
         coarse_seg_1.1_2.1
         feature0.txt
         final_map_0_seg_1.0_2.0.txt
         graph0_1.0_2.0.txt
         nodes0_1.1_2.0.txt
         time_0_seg_1.0_2.0.txt
         active_nodes_seg_1.0_2.0.txt  # the active nodes in each snapshot
         score_seg_1.0_2.0   #  the score of edges to merge in the coarsening process
 

