# Extreme Weather Event Detection

This is a project of the AI4Good course offered at ETH Zurich.

## Preparation
In order to download the needed data please run the following command:
```bash
make download-data
```
The command works on Linux/MacOS. Windows users can use WSL for to run the command.

## Running
We provide two different ways of running.

### The Easy Way
Make sure to exit from your current virtual environment and install pipenv, then run the following command:
```bash
make run-pipeline model=unet/transunet/attention/baseline stage=fit/cv
```
This command will create automatically a new virtual environment using pipenv, and runs the code within the created environment
To remove the created virtual environment, run the following command:
```bash
make remove-env
```

### The Custom Way
Install the packages in the way you prefer (we recommend to use a virtual environment in order to avoid package conflicts).
In order to run the code, execute the following the command:
```bash
python main.py
```
You can also add the following arguments to the execution of the python code:
```bash
--model=baseline/unet/attention/transunet # Select the model you wish to run
--batch_size=4 # Select the batch size
--num_workers=4 # Number of the workers for each dataloader
--small_dataset # If you want to run the code only with a portion of the dataset
--no_shuffle # If you don't want the dataloader shuffling the data
--stage=fit/cv # Fit or cross-validate the model
```
