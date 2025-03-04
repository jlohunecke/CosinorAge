<div style="display: flex; align-items: center;">
    <img src="docs/source/_static/logo.png" alt="Logo" width="150" height="150">
    <h1 style="margin-right: 10px;">CosinorAge</h1>
</div>

[![Documentation Status](https://readthedocs.org/projects/cosinorage/badge/?version=latest)](https://cosinorage.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/cosinorage.svg)](https://pypi.org/project/cosinorage/)

An open-source Python package for estimating biological age based on circadian rhythms derived from accelerometer data. The package offers a unified framework that integrates data preprocessing, feature extraction, and predictive modeling of biological age using the CosinorAge biomarker.

## Installation

```bash
pip install cosinorage
```
 To avoid issues with the installation, please make sure that the dependencies stated in the `requirements.txt` file are installed. It is recommended to set up a fresh virtual environment using Python >= 3.10.


## Package Functionalities

For a detailed example of how to use the package, please refer to the examples in the [examples](examples) folder. In addition to the below explanations, a detailed description of the package functionalities (inlcuding references to the underlying papers) can be found in the following [PDF](CosinorAge_Detailed_Package_Description.pdf). To get a detiled documentation of the package source code, please refer to the [Read the Docs](https://cosinorage.readthedocs.io/en/latest/?badge=latest).

### Modular Scheme 

![Package Data Scheme](docs/figs/schema.jpg)

### Data Loading

Depending on the data source, different data handlers are used to perform the necessary preprocessing and data conversion steps. These handlers ensure that the data is consistently formatted as minute-level ENMO data.

The figure below illustrates the supported data sources and the required file types for each DataHandler:

![Supported Data Sources](docs/figs/flowchart.png)

Below, you will find a detailed description of each supported data source and its corresponding DataHandler.

#### GalaxyDataHandler

The GalaxyDataHandler is used to load and preprocess data from the Galaxy Smartwatch. The data is expected to be located in a directory with the following structure:

![Samsung Galaxy Smartwatch Data Directory Structure](docs/figs/Smartwatch_data.png)

For each day a seperate subdirectory is expected to be present - within each day's subdirectory, the data is expected to be located in seperate hourly .binary files (file names need to start with "acceleration_data"). The binary files need to have the following 4 columns: unix_timestamp_in_ms, acceleration_x, acceleration_y, acceleration_z. The data can then be loaded into the corresponding GalaxyDataHandler object as follows.

```python
galaxy_handler = GalaxyDataHandler(gw_file_dir='../data/smartwatch/GalaxyWatch_Case1/', preprocess=True, preprocess_args=preprocess_args, verbose=True)
```

#### NHANESDataHandler

The NHANESDataHandler is used to load and preprocess data from the NHANES study. The data is expected to be located in a directory with the following structure:

![NHANES Data Directory Structure](docs/figs/NHANES_data.png)

It is expected that for a specific version of the dataset (e.g., G or H) three files are present: PAXDAY_&lt;version&gt;.xpt, PAXMIN_&lt;version&gt;.xpt and PAXHD_&lt;version&gt;.xpt. The files follow a very specific format, containing a wide range of different data fields. Thus, please use the files as they are provided by NHANES - they can be downloaded via this [link](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination). The data can then be loaded into the corresponding NHANESDataHandler object as follows.

```python
nhanes_handler = NHANESDataHandler(nhanes_file_dir='../data/nhanes/', person_id=62164, verbose=True)
```

#### UKBDataHandler

The UKBDataHandler is used to load and preprocess data from the UK Biobank. The data is expected to be located in a directory with the following structure - however, please note that the data is not publicly available:

![UKB Data Directory Structure](docs/figs/UKB_data.png)

The .csv files containing the ENMO data are expected to be in a common directory - the .csv files also contain the information needed by the UKBDataHandler to correctly determine the timestamps of the ENMO data. Below, please find an example of the content of the .csv files (made up data):

![UKB ENMO Data Example](docs/figs/UKB_csv_excerpt.png)

 In addition to that, the UKBDataHandler also expects a path to a Quality Control .csv file which contains flags for each measured ENMO series indicating whether the data is of acceptable quality. The file needs to have the following columns.

![UKB Quality Control File Columns](docs/figs/UKB_QA_cols.png)
 
 The data can then be loaded into the corresponding UKBDataHandler object as follows.

```python
ukb_handler = UKBDataHandler(qa_file_path='../data/ukb/UKB Acc Quality Control.csv', ukb_file_dir='../data/ukb/UKB Sample Data/1_raw5sec_long', eid=1000300, verbose=True)
```

### Wearable Feature Computation

The `WearableFeatures` object can be used to compute various features from the minute-level ENMO data.

```python
features = WearableFeatures(smartwatch_handler)
features.run()
```

In the following sections the different wearable features are described ion detail. To genereate all the visulaizations at once you can run the following function.

```python
dashboard(features)
```

#### Circadian Rhythm Analysis - Cosinor Analysis

The `WearableFeatures` module performs a cosinor analysis to determine the acrophase, amplitude, and MESOR of the ENMO signal over a multiday period (see below).

![Cosinor Analysis](docs/figs/cosinor_plot.png)

#### Circadian Rhythm Analysis - Non-parametric Analsis

In addition, the module performs non-parametric analyses of the circadian rhythm, generating various metrics, including L5, M10, IS, IV, and RA. For a more detailed description of these metrics, please refer to the documentation. Below, you can see an example of the visualized results.

![M10 & L5](docs/figs/M10_L5_plot.png)

![IS and IV](docs/figs/IS_IV_plot.png)

![RA](docs/figs/RA_plot.png)

#### Sleep Analyis

The module is also capable of predicting sleep phases based on the provided ENMO signal. Using the predicted sleep phases, various metrics—including TST, WASO, PTA, NWB, and SOL—are computed. Exemplary results are visualized below.

![Sleep Phases](docs/figs/sleep_plot.png)
![Sleep Metrics](docs/figs/sleep_metrics_plot.png)

#### Physical Activity Analysis 

![Physical Activity](docs/figs/PA_plot.png)

### CosinorAge Prediction

The `CosinorAge` object can be used to compute CosinorAge. It is capable of processing multiple datahandlers at the same time.

```python
records = [
    {'handler': data_handler, 
     'age': 40, 
     'gender': 'male'
    }
]

cosinor_age = CosinorAge(records)
cosinor_age.get_predictions()
```

The plot below visualizes the CosinorAge advance of a patient, showing a CosinorAge of approximately 56 years, which is higher than the patient’s chronological age of 53.

![CosinorAge Prediction](docs/figs/cosinorage.png)

## Open Source Development

The package is developed in an open-source manner. We welcome contributions to the package. 

### Clone the Repository

Clone the repository and install the package with:

```bash
git clone https://github.com/yourusername/cosinorage.git
cd cosinorage
pip install .
```

### Improve the Package

Any kind of contribution is welcome! In order to make new data sources compatible with the package, you can implement a new datahandler class - the package offers a DataHandler class that should be used as a base class.

### Implement and Execute Tests

To make sure that the changes you made are not breaking the package, you should implement new and execute the new and old tests. To do so, go to the root directory of the repository and execute the following command to run the tests:

```bash
pytest
```

Upon push to github, the tests will be executed automatically by the CI/CD pipeline and the commit will only be accepted if all the tests pass.

### Deploy Package

Build the package:
```bash
pip install build
python -m build
```

Upload the package to the PyPI repository:

```bash
pip install twine
twine upload dist/*
```

