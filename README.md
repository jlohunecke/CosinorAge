<div style="display: flex; align-items: center;">
    <img src="docs/source/_static/logo.png" alt="Logo" width="150" height="150">
    <h1 style="margin-right: 10px;">CosinorAge</h1>
</div>

[![Documentation Status](https://readthedocs.org/projects/cosinorage-deployed/badge/?version=latest)](https://cosinorage-deployed.readthedocs.io/en/latest/)


An open-source Python package for estimating biological age based on circadian rhythms derived from accelerometer data. The package offers a unified framework that integrates data preprocessing, feature extraction, and predictive modeling of biological age using the CosinorAge biomarker.

## Getting Started

### Prerequisites
- Python >= 3.9
- pip (Python package installer)
- git

### Installation Steps

#### Option 1: Quick Installation via pip (Recommended)
```bash
pip install cosinorage
```

#### Option 2: Manual Installation from Source

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/cosinorage.git
   cd cosinorage
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Create a new virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Upgrade pip
   pip install --upgrade pip

   # Install required packages
   pip install -r requirements.txt
   ```

4. **Install the Package**
   ```bash
   # Install in development mode
   pip install -e .
   ```

### Quick Start
After installation, you can start using the package:

```python
import cosinorage
from cosinorage.datahandlers import GalaxyDataHandler

# Initialize a data handler
handler = GalaxyDataHandler(galaxy_file_dir='path/to/your/data')

# Compute features
from cosinorage.features import WearableFeatures, dashboard
features = WearableFeatures(handler)
dashboard(features)

# Compute CosinorAge (dummy data)
from cosinorage.bioages import CosinorAge
records = [
    {'handler': handler, 
     'age': 60, 
     'gender': 'male', 
    }
]
predictions = CosinorAge(records)
predictions.get_predictions()
```

For more detailed examples, please refer to the [examples](examples) directory.

## Package Functionalities

For detailed examples of how to use the package, please refer to the [examples](examples) folder. The package's theoretical background, including references to the underlying research papers, is documented in our [Read the Docs theoretical background section](https://cosinorage.readthedocs.io/en/latest/theoretical_background.html). For comprehensive API documentation, please see the [package reference](https://cosinorage.readthedocs.io/en/latest/cosinorage.html).

### Modular Scheme 

![Package Data Scheme](docs/figs/schema.jpg)

### Data Loading

Depending on the data source, different data handlers are used to perform the necessary preprocessing and data conversion steps. These handlers ensure that the data is consistently formatted as minute-level ENMO data.

The figure below illustrates the supported data sources and the required file types for each DataHandler:

![Supported Data Sources](docs/figs/flowchart.png)

Below, you will find a detailed description of each supported data source and its corresponding DataHandler.

#### GalaxyDataHandler 

> ⚠️ **DISCLAIMER**: Signal scaling factors have been calibrated through internal validation experiments. Use with caution.

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

#### GenericDataHandler

The GenericDataHandler is a flexible data handler for processing various types of accelerometer and ENMO data from CSV files. It supports multiple data formats and automatically handles data preprocessing, filtering, and ENMO calculation.

The handler supports three main data types:
- **ENMO data**: Pre-calculated ENMO (Euclidean Norm Minus One) values
- **Accelerometer data**: Raw 3-axis accelerometer data (x, y, z)
- **Alternative count data**: Activity count data

Example usage for ENMO data:
```python
generic_handler = GenericDataHandler(
    file_path='data/enmo_data.csv',
    data_type='enmo',
    time_column='timestamp',
    data_columns=['enmo'],
    preprocess_args=preprocess_args,
    verbose=True
)
```

Example usage for accelerometer data:
```python
generic_handler = GenericDataHandler(
    file_path='data/accel_data.csv',
    data_type='accelerometer',
    time_column='time',
    data_columns=['x', 'y', 'z'],
    preprocess_args=preprocess_args,
    verbose=True
)
```

The GenericDataHandler automatically:
- Loads data from CSV files with customizable column names
- Detects sampling frequency from timestamps
- Filters incomplete days and selects longest consecutive sequence
- Resamples data to minute-level
- Applies preprocessing (wear detection, noise removal, etc.)
- Calculates minute-level ENMO values

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

The plot below visualizes the CosinorAge advance of a patient, showing a CosinorAge of approximately 56 years, which is higher than the patient's chronological age of 53.

![CosinorAge Prediction](docs/figs/cosinorage.png)

## Open Source Development

The package is developed in an open-source manner. We welcome contributions to the package. 

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

## Citation
Shim, J., Fleisch, E. & Barata, F. Circadian rhythm analysis using wearable-based accelerometry as a digital biomarker of aging and healthspan. npj Digit. Med. 7, 146 (2024). https://doi.org/10.1038/s41746-024-01111-x

## Acknowledgments
This project is developed in collaboration with [ADAMMA](https://adamma.ethz.ch/) - the Core for AI and Digital Biomarker Research at ETH Zurich. ADAMMA innovates at the intersection of technology, medicine, and data science to revolutionize health monitoring and disease prevention.