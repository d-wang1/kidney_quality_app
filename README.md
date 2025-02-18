# Overview
This application processes a list of kidney pathology (.svs) slides specified in a CSV file. It analyzes key image quality metrics, including staining colors, blur, and brightness, and outputs these measurements to a results file. The tool is designed to assist in assessing the quality of H&E-stained slides, specifically in quality assurance checks to ensure data quality consistency between training data and clinical data.

# Features

- Batch Processing: Reads a CSV file containing a list of SVS slide files for automated analysis.

- Staining Color Analysis: Extracts and quantifies hematoxylin and eosin (H&E) staining.

- Blur Detection: Computes a blur coefficient to evaluate image sharpness.

- Brightness Measurement: Determines overall slide brightness levels.

- CSV Output: Saves analysis results in a structured CSV format for further evaluation.

# Requirements
- Python 3.10
- Packages as specified in requirements.txt


### Install dependencies:

pip install -r requirements.txt

# Usage

Prepare a CSV file listing the SVS files to be analyzed. The CSV format should have be headerless and contain one column:

```
slide1.svs
slide2.svs
slide3.svs
```

Change the config.json. Use `source: csv`, and modify `file` to the name of the csv file specifying the file names. All the file names found in the csv file should be in the same directory, which is specified by `dir`.


### Run the analysis script:
To start calculations, run
`python main.py run`

To check whether the files are found according to the csv and configuration, use
`python main.py test-load-files`


_Note: If syntax errors arise, verify that the python version used is 3.10 via `python --version`_

The results CSV will contain the following columns:

```
File Name, Blur Coefficient, Hematoxylin RGB, Eosin RGB, Brightness, File Path
```

