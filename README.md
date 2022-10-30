# Extreme Weather Event Detection

This is a project of the AI4Good course offered at ETH Zurich.

## Preparation
Make sure to have pipenv installed.

## Pipeline
We use Makefile in order to hide the complexity of the shell commands.

### Running on a Local Machine
Here we show the steps to follow in order to run our models on a local machine, it works on both MacOS and Linux, for Windows users we'd recommend to use WSL. For simplicity, just run the following command:

```bash
make pipeline data=true/false
# If data is set to false, data won't be downloaded nor extracted.
# If you need to download and extract the data please set the variable to true.
# By default data=false.
```

In the following, the content of the "make pipeline" command is explaind in detail.

#### Data Downloading
For the dimensionality reason, the dataset is not loaded into the repository. In order to download the dataset, run the following command:

```bash
make download-data
make unzip-data # Data will be unziped to the "dataset" folder in the root folder of the Makefile
```

#### Environment Setting
In order to have assure a clean environment, without disturbances for different versions of packages, the use of a virtual environment is prefered, we used pipenv for its simplicity:

```bash
make set-pipenv
```

In order to freeze the packages installed in the virtual environment to requirements.txt, run the following:

```bash
make freeze
```

#### Run a model
Just run the following command:

```bash
make run model="the model you want" # This command still has to be completed
```

### Running on Euler
In order to run the models on Euler run the following commands:

```bash
# This command has to be run on your own machine.
make pipeline-local username="your eth username" data=true/false
# The source code is not uploaded automatically anymore, in order to upload the code to euler, please push to the repository and make a pull/clone on Euler.

# This command has to be run on Euler
make pipeline-euler data=true/false
```

The details of the pipeline here is mostly similar to the normal pipeline, including also some commands needed to upload the data to Euler. For curiosity you are free to give a look to Makefile.
