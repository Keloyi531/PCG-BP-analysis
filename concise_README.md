After three steps of selecting clean signal among the 78 participants in PCG-BP dataset, 20 participants have been selected.

### Table 1: Participants retained after the third selection based on the consistency of provisional S1/S2 detection.

| Selected Participants | | | | |
| :---: | :---: | :---: | :---: | :---: |
| 08 | 09 | 16 | 17 | 18 |
| 19 | 23 | 24 | 32 | 33 |
| 37 | 41 | 42 | 44 | 53 |
| 54 | 55 | 68 | 72 | 78 

Next, the original signals of these participants are taken from `datasets\PCG-BP\Original_signal\Original_78_participants`.
S1 and S2 are detected using `S1S2 detecion`. Based on the edge of S1 onset or S2 offset, the signals are segmented for each participant, no overlap between adjacent segments. The segments
of each participant are stored in `PCG-BP_analysis_version2\data`. Then, the participants' ID are reorganized to 1-20 rather than directly using their original ID. The data with reorganized ID are 
stored in `PCG-BP_analysis_version2\data_reorder`.

Then, **time-domain features** and **normalized amplitude features** are calculated for the segments of each participant. The features definitions are based on paper: https://onlinelibrary.wiley.com/doi/full/10.1155/2018/4205027 .

