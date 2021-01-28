# vesselness_covid
vesselness_covid is a lung vessel segmentation analysis pipeline. This program computes relevant metrics on intralung vessels with the goal of distinguishing healthy patients from the ones suffering from coronavirus. This pipelines extracts the data to a json file and provides a basic visualisation tool.

## Prior requirements
The raw data must be processed before being fed to this pipeline:
- The data must be segmented before hand ([RORPO](https://github.com/JonasLamy/LiverVesselness))
- The lung mask must also be precomputed
### Packages installation
```
pip3 -r requirements.txt
```
### Usage
#### Execution
```
python3 main.py segmented_img mask output_path
```
#### Visualisation
```
python3 visualisation.py output_path
```
