# PACE-Domain-Adaptation

## Instructions

### Step 1: Download Data
Download and unzip data into their respective folders in the `data` folder. DomainNet data can be downloaded from [here](http://ai.bu.edu/M3SDA/). Download the original version (not the cleaned version). Office-Home data can be downloaded from [here](https://www.hemanthdv.org/officeHomeDataset.html). Unzip and rename the folders such that the directory structure looks like:
```
data\
   -> office_home\
      -> Art\
      -> Clipart\
      -> Product\
      -> Real\
   -> multi\
      -> real\
      -> clipart\
      -> painting\
      -> sketch\
```
If you must download the data into a different directory (e.g. on a disk that is faster to read/write from), you can use the `--data_root` flag to indicate the location of the data as shown in the shell scripts.

### Step 2: Run PACE
We provide two shell scripts: `run_get_results_large.sh` and `run_get_results_small.sh`. These are used to reproduce the large ensemble and small ensemble results in our paper. Run either `run_get_results_large.sh` or `run_get_results_small.sh`. Do not run both; the small ensemble results are included in the large ensemble results. We tested the code on Linux OS with single P100 or V100 GPU. On a P100 GPU, the inference step takes 9.9 hours for DomainNet and 1.1 hours for Office-Home. DomainNet training takes 1.4 hours per set of results. Office-Home training takes 0.4 hours per set of results. There are two sets of DomainNet results and three sets of Office-Home results. The total time to run `run_get_results_small.sh` is 9.9 + 2 * 1.4 + 1.1 + 3 * 0.4 = 15 hours. `run_get_results_large.sh` takes 4 times as long.

## Description of Files
`data/` This directory contains the text files for the data splits to reproduce our main results. These are identical to previous SSDA work.

`get_pre_trained_weights.py` Calculates the features for each backbone. The features will be written to the `feature_weights` folder.

`get_results.py` Trains a linear classifier for SSDA. Prints results. Saves results as a dictionary in `*.dic`.

`get_results_UDA.py` Trains a linear classifier for UDA. Prints results. Saves results as a dictionary in `*.dic`.

`get_results_large.py` Assumes `get_results.py` has been run for each of the 4 augmentation settings. Reads in the results from `*.dic` and prints results for the large ensemble. This file takes under a minute to run.

`get_results_large_UDA.py` Assumes `get_results_UDA.py` has been run for each of the 4 augmentation settings. Reads in the results from `*.dic` and prints results for the large ensemble. This file takes under a minute to run.

## Sample Outputs
Sample output from `get_results_UDA.py`:
```
(augmentation, shots):  ('none', 0)
================================================================================
Results for each ensemble member
================================================================================
Net, R->C, R->P, R->A, P->R, P->C, P->A, A->P, A->C, A->R, C->R, C->A, C->P, 
convnext_xlarge_384_in22ft1k, 81.83276057243347, 95.04392743110657, 90.15244841575623, 94.14734840393066, 79.47308421134949, 87.88627982139587, 93.33183169364929, 80.73310852050781, 93.29813718795776, 92.93091297149658, 89.57560658454895, 93.33183169364929, 
convnext_xlarge_in22ft1k, 82.42840766906738, 94.79612112045288, 88.79274725914001, 94.0096378326416, 81.07674717903137, 87.18582391738892, 93.39941143989563, 80.34364581108093, 93.64241361618042, 93.61945986747742, 88.46312165260315, 93.44446659088135, 
convnext_xlarge_in22k, 82.42840766906738, 94.66095566749573, 89.16357755661011, 93.9866840839386, 79.86254692077637, 88.0510926246643, 91.5521502494812, 79.93127107620239, 93.68831515312195, 93.3669924736023, 88.29830884933472, 93.64721775054932, 
swin_large_patch4_window7_224, 80.84765076637268, 94.14282441139221, 87.88627982139587, 93.9866840839386, 80.1374614238739, 86.320561170578, 92.0477569103241, 80.57274222373962, 93.82602572441101, 93.34404468536377, 87.14461922645569, 91.79995059967041, 
swin_large_patch4_window7_224_in22k, 81.74112439155579, 94.23293471336365, 87.35063672065735, 93.96373629570007, 78.87743711471558, 86.19694709777832, 92.94885993003845, 79.67926859855652, 93.84897947311401, 92.63254404067993, 87.22702860832214, 91.64226055145264, 
swin_large_patch4_window12_384, 82.61168599128723, 94.3455696105957, 88.0510926246643, 94.23915147781372, 81.3516616821289, 87.80387043952942, 93.26424598693848, 81.42039179801941, 93.41289401054382, 93.27518939971924, 89.08116817474365, 91.98017120361328, 
swin_large_patch4_window12_384_in22k, 82.15349316596985, 94.05271410942078, 88.25710415840149, 93.82602572441101, 79.15235161781311, 86.8149995803833, 92.70105957984924, 81.78694248199463, 94.0096378326416, 93.55060458183289, 88.83395195007324, 92.47578382492065, 

================================================================================
Simple Majority vote
================================================================================
R->C, R->P, R->A, P->R, P->C, P->A, A->P, A->C, A->R, C->R, C->A, C->P, 
84.05498266220093, 95.04392743110657, 89.86402750015259, 94.67523694038391, 82.63459205627441, 88.83395195007324, 93.96260380744934, 82.42840766906738, 94.42276954650879, 94.26210522651672, 90.23485779762268, 93.42194199562073, 

================================================================================
Simple Average (Usually better)
================================================================================
R->C, R->P, R->A, P->R, P->C, P->A, A->P, A->C, A->R, C->R, C->A, C->P, 
84.44444537162781, 95.134037733078, 89.94643688201904, 94.69818472862244, 82.65750408172607, 89.0399694442749, 94.07524466514587, 82.47422575950623, 94.44571733474731, 94.2162036895752, 90.11124968528748, 93.51205229759216,
```
Sample output from `get_results_large_UDA.py`:
```
================================================================================
Ask each 4x7 enasemble members to vote. Majority wins.
================================================================================
R->C, R->P, R->A, P->R, P->C, P->A, A->P, A->C, A->R, C->R, C->A, C->P, 
85.45246124267578, 95.17909288406372, 90.1524543762207, 94.72113847732544, 82.93241858482361, 89.45199847221375, 93.73732805252075, 83.52806568145752, 94.33096051216125, 94.35391426086426, 90.64688682556152, 93.44446659088135, 

================================================================================
Average prediction over 4 x 7 models.
================================================================================
R->C, R->P, R->A, P->R, P->C, P->A, A->P, A->C, A->R, C->R, C->A, C->P, 
85.58992147445679, 95.17909288406372, 90.19365310668945, 94.69818472862244, 82.97823667526245, 89.20478224754333, 94.00765895843506, 83.64261388778687, 94.35391426086426, 94.28505897521973, 90.7705008983612, 93.73732805252075, 
```

## Additional Remarks

For faster experimentation, we recommend decreasing the number of backbones in the ensemble. You can do this by editing this line:
```python
# (network, feature_size, batch_size, crop_size)
networks = [('convnext_xlarge_384_in22ft1k', 2048, 12, 384),
 ('convnext_xlarge_in22ft1k', 2048, 12, 224),
 ('convnext_xlarge_in22k', 2048, 12, 224),
 ('swin_large_patch4_window7_224', 1536, 12, 224),
 ('swin_large_patch4_window7_224_in22k', 1536, 12, 224),
 ('swin_large_patch4_window12_384', 1536, 12, 384),
 ('swin_large_patch4_window12_384_in22k', 1536, 12, 384)]
```
The backbones with 384 input resolution are the most time consuming.
