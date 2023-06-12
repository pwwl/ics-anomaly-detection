This dataset contains a set of readings across 123 features (sensors, actuators) for a water distribution system.

Request and download the WADI dataset from: 
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

- This code is currently based on the first WADI dataset (WADI.A1_9 Oct 2017).
- From the first WADI dataset, download the following two files into this directory:
    - `WADI_attackdata.csv` and `WADI_14days.csv`
- Run `process_WADI.py`, which will relabel the files as training/test CSVs:
```sh
python3 process_WADI.py
```