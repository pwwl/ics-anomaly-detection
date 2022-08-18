This dataset contains processing for a 6 stage water treatment process, collected from a water plant testbed in Singapore.
Contains 77 sensors/actuators, and 6 labelled cyber-attacks.

Request and download the SWAT dataset from: 
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

- This code is currently based on the 1st SWAT dataset (December 2015).
- Open each of the excel sheets and save them as CSVs
    - Save the corresponding files as `SWAT_Normal_v0.csv`, `SWAT_Normal_v1.csv`, and `SWAT_Attack_v0.csv`
- Run `process_SWAT.py`, which will relabel the files as training/test CSVs.