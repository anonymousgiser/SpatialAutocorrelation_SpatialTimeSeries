The structure of the folder named "STS_SA" is as follows:
STS_SA
|---datasets
|  |---real_World_STS_Data.csv
|  |---SWM_for_real_world_dataset.csv
|  |---synthetic_STS_Data.csv
|---results
|  |---global_SA_for_real_world_STS.csv
|  |---global_SA_for_synthetic_STS.csv
|  |---local_SA_for_real_world_STS.csv
|  |---local_SA_for_synthetic_STS_p_values.csv
|  |---local_SA_for_synthetic_STS_z_scores.csv
|  |---local_SA_for_synthetic_STS.csv
|---ERP.py
|---global_STS_SA_for_synthetic_dataset.py
|---local_STS_SA_for_synthetic_dataset.py
|---STS_SA_for_real_world_dataset.py
|---readme.txt

In the folder named "datasets", there are STS data for synthetic dataset and real world dataset, and the spatial weight matrix of real world dataset is also given.
----synthetic_STS_Data.csv
    In this csv file, STS data of synthetic dataset is provided in 588 rows and 10 columns. Each row corresponds to an STS, and the i-th row is the STS whose ID = i. Each column corresponds to a moment.
----real_World_STS_Data.csv
    In this csv file, STS data of real-world dataset is provided in 170 rows and 365 columns. Each row corresponds to an STS, and the i-th row is the STS whose ID = i. Each column corresponds to a day in 2018. The value is the average concentration of PM2.5 in coresponding area.
----SWM_for_real_world_dataset.csv
    The spatial weight matrix of the 170 areas in real-world dataset are stored in this file. Here we use binary weight to describe the spatial relationship of Rook's contiguity in these STSs. If two STSs are adjacent, the weight will be 1.  And the weight will be 0 if two STSs are not adjacent.



In the folder named "results", the analysis results of spatial autocorrelation for STS are provided.
----global_SA_for_synthetic_STS.csv
    The global spatial autocorrelation for syntectic dataset. The values of global indicator, Z-scores of the results, and P-values are given respectively.
----local_SA_for_synthetic_STS.csv
----local_SA_for_synthetic_STS_z_scores.csv
----local_SA_for_synthetic_STS_p_values.csv
    The local spatial autocorrelation for syntectic dataset. The values of local indicator, Z-scores of the results, and P-values are given respectively. To better express the distribution of local spatial autocorrelation, each numerical value is placed on its corresponding position.
----global_SA_for_real_world_STS.csv
    The global spatial autocorrelation for real-world dataset. The value of global indicator, Z-score of the results, and P-value are given respectively.
----local_SA_for_real_world_STS.csv
    The local spatial autocorrelation for real-world dataset. The STS ID, values of local indicator, Z-scores of the results, and P-values are given respectively.


What needs to be emphasised is that the ERP algorithm in ERP.py comes from https://github.com/bguillouet/traj-dist/blob/master/traj_dist/cydist/erp.pyx, and we modified the code according to our need. 
To run the codes successfully, some Python packages including numpy, numba, math, and scipy are required.