---
layout: page
title: Software Setup
permalink: /setup-instructions/
---

For ECE4602, the recommended way to work on assignments is through [Google Colaboratory](https://colab.research.google.com/). However, if you already own GPU-backed hardware (or just prefer local dev), you can also work locally by setting up a virtual environment.

- [Working remotely on Google Colaboratory](#working-remotely-on-google-colaboratory)
- [Working locally on your machine](#working-locally-on-your-machine)
  - [Anaconda virtual environment](#anaconda-virtual-environment)
  - [Python venv](#python-venv)
  - [Installing packages](#installing-packages)

### Working remotely on Google Colaboratory

Google Colaboratory is basically a combination of Jupyter notebook and Google Drive. It runs entirely in the cloud and comes
preinstalled with many packages (e.g. PyTorch and Tensorflow) so everyone has access to the same
dependencies. Even cooler is the fact that Colab benefits from free access to hardware accelerators
like GPUs (e.g. T4, L4, A100 — availability varies) and TPUs which will be particularly useful for assignments 2 and 3.

**Requirements**. You need a Google account (so Colab can save notebooks to Google Drive).

**Workflow**.
1. Download the assignment starter zip from the course website.
2. Unzip it on your computer.
3. Upload the unzipped folder to Google Drive.
4. In Colab, use `File -> Open notebook -> Google Drive` and open the `.ipynb` files from your uploaded folder.

**Best Practices**. There are a few things you should be aware of when working with Colab. The first thing to note is that resources aren't guaranteed (this is the price for being free). If you are idle for a certain amount of time or your total connection time exceeds the maximum allowed time (~12 hours), the Colab VM will disconnect. This means any unsaved progress will be lost. <font color="red"><strong>Thus, get into the habit of frequently saving your code whilst working on assignments.</strong></font> To read more about resource limitations in Colab, read their FAQ [here](https://research.google.com/colaboratory/faq.html).

**Using a GPU**. Using a GPU is as simple as switching the runtime in Colab. Specifically, click `Runtime -> Change runtime type -> Hardware Accelerator -> GPU` and your Colab instance will automatically be backed by GPU compute.

If you're interested in learning more about Colab, we encourage you to visit the resources below:

* [Intro to Google Colab](https://www.youtube.com/watch?v=inN8seMm7UI)
* [Welcome to Colab](https://colab.research.google.com/notebooks/intro.ipynb)
* [Overview of Colab Features](https://colab.research.google.com/notebooks/basic_features_overview.ipynb)

### Working locally on your machine
If you wish to work locally, you should use a virtual environment. You can install one via Anaconda (recommended) or via Python's native `venv` module. Use Python 3.10+ (Python 3.11 is a good default); Python 2 is not supported.

#### Anaconda virtual environment
We strongly recommend using the free [Anaconda Python distribution](https://www.anaconda.com/download/) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), which provides an easy way to handle package dependencies.

Once you have Anaconda/Miniconda installed, create a virtual environment for the course. To set up a virtual environment called `ece4602`, run:

```bash
# this will create an anaconda environment
# called ece4602 in '.../anaconda3/envs/'
conda create -n ece4602 python=3.11
```

To activate and enter the environment, run `conda activate ece4602`. To deactivate the environment, run `conda deactivate`. Note that every time you want to work on an assignment, you should activate the environment first.

```bash
# sanity check that the path to the python
# binary matches that of the anaconda env
# after you activate it
which python
# (Windows PowerShell) Get-Command python
# (Windows cmd) where python
```

You may refer to [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more detailed instructions on managing virtual environments with Anaconda.

**Note:** If you've chosen to go the Anaconda route, you can safely skip the next section and move straight to [Installing Packages](#installing-packages).

<a name='venv'></a>
#### Python venv

As of 3.3, Python natively ships with a lightweight virtual environment module called [venv](https://docs.python.org/3/library/venv.html). Each virtual environment packages its own independent set of installed Python packages that are isolated from system-wide Python packages and runs a Python version that matches that of the binary that was used to create it.

```bash
# this will create a virtual environment
# called .venv in the current directory
python -m venv .venv
```

To activate it:
- macOS/Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

To deactivate the environment, run `deactivate`.

```bash
# sanity check that the path to the python
# binary matches that of the virtual env
# after you activate it
which python
```

<a name='packages'></a>
#### Installing packages

Once you've **setup** and **activated** your virtual environment (via `conda` or `venv`), `cd` into the unzipped assignment folder (for example `assignment1/`) and install any needed libraries using `pip`:

```bash
# upgrade packaging tools
python -m pip install --upgrade pip

# if the assignment folder includes a requirements file, install it
if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; fi

# if you want to run notebooks locally
python -m pip install notebook
```

On Windows PowerShell, the `requirements.txt` line becomes:

```powershell
if (Test-Path requirements.txt) { python -m pip install -r requirements.txt }
```
