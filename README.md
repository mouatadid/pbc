# Probabilistic Bias Correction

Probabilistic bias correction (PBC) is a machine learning framework for improving subseasonal forecasts from dynamical, AI, or hybrid prediction systems. Given a raw ensemble of deterministic forecasts, PBC outputs learning-enhanced probabilities designed to correct systematic errors while preserving the predictive signal in the underlying model. 

To learn more, see the accompanying study:

[Enhancing AI and Dynamical Subseasonal Forecasts with Probabilistic Bias Correction](https://arxiv.org)

```bib
@article{guan2026enhancing,
  author = {Hannah Guan and Soukayna Mouatadid and Paulo Orenstein and Judah Cohen and Haiyu Dong and Zekun Ni and Jeremy Berman and Genevieve Flaspohler and Alex Lu and Jakob Schloer and Joshua Talib and Jonathan A. Weyn and Lester Mackey},
  title = {Enhancing AI and Dynamical Subseasonal Forecasts with Probabilistic Bias Correction},
  journal = {arXiv preprint},
  year = {2026}
}
```

## System Requirements and Recommendations

This codebase has been tested with the following operating system and Python pairings:
+ Rocky Linux 8.10 with Python 3.13.3

A complete list of Python dependencies can be found in `src/setup/environment-aiwqd.yml`.

## Getting Started

- Add the `pbc/src` directory to your `PYTHONPATH` to ensure Python can import our code modules.
  - Add `export PYTHONPATH=/home/$USER/pbc/src:$PYTHONPATH` to your `~/.bashrc` file (or equivalent shell initialization script) and open a new shell.
- Create and activate a conda environment with all dependencies installed
  ```bash
  conda env create -f src/setup/environment-aiwqd.yml
  conda activate aiwqd
  ```
  - This installation completed in under 3 minutes on a Rocky Linux 8.10 machine with 80 GB of RAM and 8 cores.
- Clone this repository and run all scripts from the repository base directory (this directory)
- Run the following demo which generates global first-quintile Climatology forecasts for mean sea level pressure over the 2016-2024 `std_test` evaluation period with a 19-day lead time
  `python src/models/climatology/batch_predict.py era5-f1_mslp 19 -t std_test`
  - This demo ran to completion in 13 seconds with Python 3.13.3 on a Rocky Linux 8.10 machine with 80 GB of RAM and 8 cores.
  - Expected outputs 
    - A forecast folder `models/climatology/submodel_forecasts/climatology/era5-f1_mslp_19/` containing biweekly forecast files from 20160101 through 20241230

## Generating Model Forecasts

The following examples demonstrate how to generate PBC-ECMWF forecasts for the 2016-2024 `std_test` evaluation period in "Enhancing AI and Dynamical Subseasonal Forecasts with Probabilistic Bias Correction".

- Debias++-ECMWF:
  ```bash
  # Form predictions for each quintile
  for k in 1 2 3 4; do
    # First generate and evaluate predictions for each model configuration
    python src/models/ecmwfpp/bulk_batch_predict.py era5-f{k}_mslp 19 -t std_tune -wm
    # Then select a model configuration using the tuner
    python src/models/tuner/batch_predict.py era5-f{k}_mslp 19 -mn ecmwfpp -t std_test
  done
  # Project forecasts onto the space of valid distributions
  python src/models/projector/batch_predict.py era5-mslp 19 -mn tuned_ecmwfpp -t std_test
  ```
- Persistence++-ECMWF:
  ```bash
  # Form predictions for each quintile
  for k in 1 2 3 4; do
    python src/models/perpp_ecmwf/bulk_batch_predict.py era5-f{k}_mslp 19 -t std_test
  done
  # Project forecasts onto the space of valid distributions
  python src/models/projector/batch_predict.py era5-mslp 19 -mn perpp_ecmwf -t std_test
  ```
- PBC-ECMWF:
  - After generating the corresponding Debias++-ECMWF and Persistence++-ECMWF forecasts, 
  ```bash
  # Form predictions for each quintile
  for k in 1 2 3 4; do
    python src/models/abc_ecmwf/bulk_batch_predict.py era5-f{k}_mslp 19 -t std_test
  done
  ```

## Downloading and Processing Data

The `src/data` folder contains the source code to download, process, and store relevant data files. 
When running these scripts, you will want to activate the `aiwqd` conda environment.

### Downloading ERA5 data

To download ERA5 data from `cdsapi`, please follow step 1 ("Setup the CDS API personal access token") in this [link](https://cds.climate.copernicus.eu/how-to-api), resulting in the file `~/.cdsapirc`, which will automatically be used by our ERA5 download scripts to authenticate with the `cdsapi` library.

### Downloading ECMWF forecast data

Complete the following authentication steps to download ECMWF forecast data from the IRI data library.

- Sign up for a free IRI account at https://iridl.ldeo.columbia.edu/auth/signup
- Accept the license and terms agreement in https://iridl.ldeo.columbia.edu/SOURCES/.ECMWF/.S2S/.ECMF/.CY41-47/.forecast/.control/.2m_above_ground/.2t/datatables.html
- `conda create -n pycpt-2.9.3 --file src/setup/conda-linux-64.lock`
- `conda activate pycpt-2.9.3`
- `python`
- At the python prompt,
    - `import cptdl; cptdl.setup_dlauth("your.email@address.com")`
    - Provide your Data Library password when prompted.
    - `exit()`
- You will now have a `~/.pycpt_dlauth` file, which will automatically be used by our ECMWF download scripts to authenticate with the IRI data library.

Complete the following authentication steps to download backup ECMWF forecast data from the ECMWF servers.

- Register with ECMWF at https://www.ecmwf.int
- Retrieve you API access key at https://api.ecmwf.int/v1/key/
- Copy and paste the API access key into the file `~/.ecmwfapirc`

