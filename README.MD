# Intro
This repository contains code for experiments using the winning mode for xView2  "xView2: Assess Building Damage" challenge and a simple implementation of our strong baseline. 
You will find in ./legacy all the code and results of an ablation study on the winning model. A second README is available in ./legacy. The rest of the repo is dedicated to an implementation of the strong baseline using Pytorch Lightning, Hydra and WandB.

# Architecture
This repo is organized to work using Pytorch Lightning and hydra :
```
.
├── __init__.py
├── README.MD
├── results.MD
├── requirements.txt
├── main.py
├── datasets
│   ├── base_dataset.py
│   ├── supervised_dataset.py
│   ├── transforms.py
├── modules
│   ├── supervised.py
├── sbatchs
│   ├── main_supervised.sbatch
├── conf
│   ├── data
│   ├── network
│   ├── supervised_config.yaml
│   └── trainer
├── legacy
|   ├── README.MD
│   ...
|   ...
├── logs
│   ...
└── wip
    ...
```

- `main.py` is the main file for the ssl experiments. It is used to launch the experiments. It uses the hydra and config.yaml file in ./conf to set the parameters of the experiments. main.py is meant to be launched with the m main_supervised.sbatch in ./sbatchs depending on the experiment you want to run (ssl or supervised baseline). Note that the supervised baseline is a rework of the ablated model using pytorch lightning.
- `./datasets` contains the dataset classes used in the experiments. The base_dataset.py is the base class for all the datasets. The supervised_dataset.py has the  the lightning datamodule class used for the supervised baseline. transforms.py contains the custom transforms used for the datasets.
- `./modules` contains the model classes. supervised.py contains the model for the strong baseline. It is based on the legacy resnet model defined in ./legacy/zoo/models.py.
- `./sbatchs` contains the sbatch files used to launch the experiments on the cluster.
- `./conf` contains the hydra config files used to set the parameters of the experiments.
- `./legacy` contains the code and results of the ablation study on the winning model. A separate README is available in ./legacy.
- `./logs` contains the logs of the experiments.
- `./wip` contains some work in progress code that was not used in the final experiments and can be ignored or deleted.


# How to use
## Installation
### Requirements
we are using Python 3.8.2. For the required packages, see the requirements.txt file. You can install them using the following command:
```pip install -r requirements.txt```

this might not work as your hardware might require another version of cuda but you can still read the requirements.txt file and install the packages you need manually. For our hardware we used the following command:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade```
### Setting up the dataset

The dataset is available on the [xView2 challenge website](https://xview2.org/). Be sure to download the train, test and tier3 sets.

The you will have to create the masks to be used during training as ground truth (in the data you have just downloaded you will find target masks that are also ground truth but are not formatted correctly (they are multiclass and not multilabel gt masks)). To do so, you will have to modify the `./legacy/create_masks.py` script (specify the train_dirs for which you want to create the masks /!\ we recommend to create masks for train, test and tier3 sets /!\ ) and launch it using the following command:

```python ./legacy/create_masks.py```

Finally be sure to modify the arguments marked as ####To be changed#### in the `./config` files. `data.data_dirs` should be the 3 directories for which you just created masks. `model_dir` should be a location where you are comfortable with storing model weights and predictions


## Launching the experiments 

This repo has been designed to be pretty straightforward to use once set up. Modify the config files in `./config` to set the parameters of the experiments and launch it using the main.py file. The main.py file is meant to be launched using the sbatch files in `./sbatchs` For example using the following command from thee root of the repo:

```sbatch ./sbatchs/main_supervised.sbatch ```

if your are using different hardware / cluster framework, consider modifying the `./conf/trainer/supervised_trainer.yaml` file accordingly and launch the experiments using the following command:

```python /Midgard/home/paulbp/xview2-replication/main.py --config-name supervised_config group="name_of_the_wandb_group"```

the arguments from supervised_trainer.yaml can be overwritten using the command line. (see the sbatch file).

Once the experiments are launched, you can monitor them using wandb. The results are saved in `./logs`.
