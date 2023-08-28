# ICS Anomaly Detection Test Suite

This repository contains code related to "[Perspectives from a Comprehensive Evaluation of Reconstruction-based Anomaly Detection in Industrial Control Systems](https://www.ece.cmu.edu/~lbauer/papers/2022/esorics2022-ics-anomaly-detection.pdf)", 
presented and published at [ESORICS 2022](https://esorics2022.compute.dtu.dk/). 

## Cite This Work

    @inproceedings{fung22-ics-anomaly-detection,
      title =        {Perspectives from a comprehensive evaluation of reconstruction-based anomaly detection in industrial control systems},
      author =       {Clement Fung and Shreya Srinarasi and Keane Lucas and Hay Bryan Phee and Lujo Bauer},
      booktitle =    {ESORICS 2022: 27th European Symposium on Research in Computer Security},
      url =          {https://www.ece.cmu.edu/~lbauer/papers/2022/esorics2022-ics-anomaly-detection.pdf},
      year =         {2022},
    }

## Code

### Installation

This project uses Python3 and Tensorflow, which requires 64-bit Python 3.8-3.11.
For compatibility with required packages, we recommend using any installment of 64-bit Python 3.8.x. 
The best way to get set up quickly is through a Python virtual environment (like virtualenv).
Here is [a detailed guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#installing-packages-using-pip-and-virtual-environments) to installing virtualenv through pip, and using virtualenv to setup a Python virtual environment.
We recommend using virtualenv and not venv so that a virtual environment with a specific Python version can be created, as shown below.

For Unix/macOS users to start up a virtual environment and activate it:  
```sh
virtualenv -p python3.8 venv  
source venv/bin/activate
```
Importantly, be sure to specify the Python version with the `-p` flag.
Note: In order to create a Python virtual environment of a specific version, the host environment must also have that specific version installed.

Once in the virtual environment, install the needed requirements:
```sh
pip install -r requirements.txt
```

### Data Setup

This repository is configured for three datasets: `BATADAL`, `SWAT`, and `WADI`.

For convenience, the BATADAL dataset is included as a `tar.gz` file. 
The raw SWaT and WADI datasets need to be requested through the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

For instructions on how to setup and process the raw datasets, see the associated README files in the `data` directory.

#### Dataset Cleaning

Some recent experiments and prior work have suggested using the Kolmogorov-Smirnov test to filter out and remove features whose train-test distributions vary significantly. This technique was proposed for the SWAT dataset in Section V of this ArXiv paper and has been locally implemented in `main_data_cleaning.py`.

To use the cleaned versions of the `SWAT` and `WADI` datasets, we have added dataset names `SWAT-CLEAN` and `WADI-CLEAN`. These will remove the features specified by the Kolmogorov-Smirnov test from the processed `SWAT` and `WADI` datasets. Specify these new dataset names when training, evaluating, and tuning models, as seen below.

### Example Workflow 

There are three main scripts:
- `main_train.py` trains anomaly detection models.
- `main_eval.py` evaluates anomaly detection models.
- `main_model_tuning.py` performs threshold tuning based on a given metric.

#### Project Setup

Each of the above scripts uses the argument `--run_name` as a tag for experiments. This ensures that files are not written over when repeating experiments with the same parameters. Each tag must have an associated subdirectory named in the `outputs`, `plots`, and `models` directories. A helper script `setup_run_name.sh` is provided for easy setup.

Example usage to generate directories named `example1`:
```sh
bash setup_run_name.sh example1
```

#### Training the model

Example of basic usage: 
```sh
python3 main_train.py AE BATADAL --run_name example1 --ae_model_params_layers 3 --ae_model_params_cf 3
```

Running the above command will train an autoencoder on the BATADAL dataset, with 3 layers and a compression factor of 3.  
For a full list of available commands, use the `--help` argument.

#### Evaluating the model

Example of basic usage: 
```sh
python3 main_eval.py AE BATADAL --run_name example1 --ae_model_params_layers 3 --ae_model_params_cf 3 --detect_params_windows 1 3 5 10 --detect_params_percentile 0.95 0.99 0.995
```

Running the above command will tune the threshold (by calculating the percentile error on the validation dataset) and window length for the previously trained autoencoder over the given set of values.  
In this example, each combination of window size and percentile will be compared (12 configurations).  
For a full list of available commands, use the `--help` argument.

#### Tuning the model

Example of basic usage:
```sh
python3 main_model_tuning.py AE BATADAL --run_name example1 --ae_model_params_layers 3 --ae_model_params_cf 3 --detect_params_hp_metrics F1 SF1 SFB13 SFB31 --detect_params_eval_metrics F1 SF1 SFB13 SFB31
```
Running the above command will perform the same above tuning, but for each of the metrics listed after `--detect_params_hp_metrics`. After each tuning is performed, the resulting model tuning will be scored on each metric listed after `--detect_params_eval_metrics`.

For a full list of available metrics and their names, see `metrics.py`.  
For a full list of available commands, use the `--help` argument.




