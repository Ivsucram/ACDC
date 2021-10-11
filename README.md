# Reference

## Paper

ACDC: Online Unsupervised Cross-Domain Adaptation

[ArXiv](https://arxiv.org/abs/2110.01326)

[Research Gate](https://www.researchgate.net/publication/355060706_ACDC_Online_Unsupervised_Cross-Domain_Adaptation)

## Bibtex

```
@misc{decarvalho2021acdc,
      title={ACDC: Online Unsupervised Cross-Domain Adaptation}, 
      author={Marcus de Carvalho and Mahardhika Pratama and Jie Zhang and Edward Yapp},
      year={2021},
      eprint={2110.01326},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

# Welcome to ACDC!

This is a framework aiming autonomously cross-domain conversion (ACDC) which handles fitting/training and concept drifts with a complete self-evolving structure, achieving domain adaptation via a domain-adversarial classifier module, all without the need for hyper-parameter tunability.

The paper is still under review. You will find the following on this repository:

- Original ACDC source-code, so you can re-validate the results presented on screen but without visualize the code.
- A compilation of ACDC numerical results, including both experiments and ablation study.
- Source-code of the baselines used throughout the paper.
- A compilation of baselines numerical results.


# Setting up your environments

We have source-codes used mainly in three languages:

- Python (including Python 2 and Python 3)
- Matlab
- Java

You will need Matlab to run the following baselines:

- ATL

You will need Python to run the following baselines:

- ACDC
- MSC
- FUSION

You will need Java (>13) to run the following baselines:

- Melanie


While Matlab source-codes are probably a plug-and-play after you install Matlab in your machine, Python source-codes will use different environments. However, we organized and configured it for you, so you can install it with a single command. Java codes are a bit more harder to handle, so the better is to follow the [original Melanie repository]([https://github.com/nino2222/Melanie]) to configure your environment. You can still use ACDC to prepare the datasets in the Melanie format.

Make sure that you have [Anaconda]([https://www.anaconda.com/](https://www.anaconda.com/)) or [Conda]([https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)) installed in your Machine. It can be Windows, Mac or Linux operational system.

Open your Anaconda Prompt and travel to the directory of the source-code you want to execute, example, ACDC directory.

Run the following command:

```conda env create -f environment.yml```

This command will create a conda enviroment called `acdc`, if you run it on the ACDC folder. The environments will automatically install the correct Python version that source-code needs (ACDC uses the most recently) and its dependencies.
If you run the above command at the MSC folder, you will install a conda environment called `msc`. The same behavior extends to the FUSION and DFAMCD folders.

# Downloading the benchmarks
To make the process simpler and automatically, all benchmarks are manage through a Python implementation. Some benchmarks are very big and heavy, so make sure you have enough storage space in your machine, while are connected to a internet connection.

ACDC will download and configure every benchmark automatically, applying concept drifts whenever necessary. If you use the `prepare_datasets.py` files found on every baseline folder, it will download and generate datasets according to what is used throughout the paper. You can read and evaluate this file to make sure the benchmarks are configured correctly.

If you want to test other variations of concept drifts, or even download and set the benchmarks without concept drifts, you can perform the following actions:

- Set up ACDC environment
- Activate ACDC conda environment
- Run `python ACDC.py` command

This command will print a number of instructions of how ACDC works, including how to download, prepare and save different benchmarks. You can also re-run ACDC with configurations similar or different from the paper.

Make sure that ACDC already downloaded every benchmark before run `prepare_datasets.py` on the baselines, as the later will use the generated `data` folder from ACDC, by executing the following command on the ACDC folder:

```
conda activate acdc
python -c "import ACDC as acdc; acdc.pre_download_benchmarks()"
```

# Running ACDC
After setup your environment, just run `python ACDC.pyc`. The script will print a list of commands for you.

## Example: Running ACDC with USPS --> MNIST experiment

After setup your environment, just run the following command in the ACDC directory:

```
python -c "import ACDC as acdc; acdc.acdc('usps-16','mnist-16',5,7,False)" 
```

or just:

```
python -c "import ACDC as acdc; acdc.acdc('usps-16','mnist-16')"
```

You can also create a Jupyter file into ACDC directory and create a cell with the following command:
```
import ACDC as acdc
acdc.acdc('usps-16','mnist-16')
```

## Example: Running ACDC Ablation Study A

The Ablation studies turn-off or disable some funcionalities from the ACDC framework. To run ACDC Ablation Study A, we would just execute the same command as before, but replacing `ACDC.pyc` by `ACDC_Ablation_A.pyc`, like:

```
python -c "import ACDC_Ablation_A as acdc; acdc.acdc('usps-16','mnist-16')
```
You can do something similar in a Jupyter file:
```
import ACDC_Ablation_A as acdc
acdc.acdc('usps-16','mnist-16')
```
