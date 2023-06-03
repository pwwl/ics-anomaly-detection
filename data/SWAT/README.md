This dataset contains processing for a 6 stage water treatment process, collected from a water plant testbed in Singapore.
Contains 77 sensors/actuators, and 6 labelled cyber-attacks.

Request and download the SWAT dataset from: 
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

- This code is currently only based on the first SWAT dataset (SWaT.A1 & A2_Dec 2015).
- From the first SWAT dataset, access the `Physical` directory to find the following three files:
    - `SwaT_Dataset_Normal_v1.xlsx`, `SwaT_Dataset_Normal_v0.xlsx`, and `SwaT_Dataset_Attack_v0.xlsx`
- Save these files as `SWAT_Normal_v0.csv`, `SWAT_Normal_v1.csv`, and `SWAT_Attack_v0.csv` respectively, in this directory.
- Run `process_SWAT.py`, which will relabel the files as training/test CSVs:
```sh
python3 process_SWAT.py
```