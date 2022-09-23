# ICS Anomaly Detection Test Suite

This repository contains code related to "Perspectives from a Comprehensive Evaluation of Reconstruction-based Anomaly Detection in Industrial Control Systems", to appear in [ESORICS 2022](https://esorics2022.compute.dtu.dk/). 

## Cite this work

    @inproceedings{fung22-ics-anomaly-detection,
      title =        {Perspectives from a comprehensive evaluation of reconstruction-based anomaly detection in industrial control systems},
      author =       {Clement Fung and Shreya Srinarasi and Keane Lucas and Hay Bryan Phee and Lujo Bauer},
      booktitle =    {ESORICS 2022: 27th European Symposium on Research in Computer Security},
      url =          {https://www.ece.cmu.edu/~lbauer/papers/2022/esorics2022-ics-anomaly-detection.pdf},
      year =         {2022},
    }

## Code

### Installation

This project uses python3 and tensorflow.
The best way to get set up quickly is through a python virtual environment (like virtualenv).

Start up a virtual environment and activate it:
`virtualenv -p python3 venv`
`source venv/bin/activate`

And install the needed requirements.
`pip install -r requirements.txt`

### Data Setup

This repository is configured for three datasets: BATADAL, SWaT, and WADI.

For convienience, the BATADAL dataset is included as a `tar.gz` file. 
The SWaT and WADI datasets need to be requested through the [iTrust website](https://itrust.sutd.edu.sg/itrust-labs_datasets/).

For instructions on how to setup and process the raw datasets, see the associated READMEs in the `data` directory.

### Running

There are three main scripts:
`main_train.py` trains anomaly detection models.
`main_eval.py` evaluates anomaly detection models.
`main_model_tuning.py` performs threshold tuning based on a given metric.


