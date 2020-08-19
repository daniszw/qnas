# Q-NAS
## Neural Architecture Search using the Q-NAS algorithm and Tensorflow.

This repository contains code for the works presented in the following papers:

>1. D. Szwarcman, D. Civitarese and M. Vellasco, "Quantum-Inspired Neural Architecture Search,"   
2019 International Joint Conference on Neural Networks (IJCNN), Budapest, Hungary, 2019, pp. 1-8.
[DOI](https://doi.org/10.1109/IJCNN.2019.8852453)
>
>2. D. Szwarcman, D. Civitarese and M. Vellasco, "Q-NAS Revisited: Exploring Evolution Fitness to Improve Efficiency,"    
2019 8th Brazilian Conference on Intelligent Systems (BRACIS), Salvador, Brazil, 2019, pp. 509-514.
[DOI](https://doi.org/10.1109/BRACIS.2019.00095)


### Requirements

The required python packages are listed in `requirements.txt`.  
Make sure you have `openmpi` (https://www.open-mpi.org/) installed before installing the project requirements.   
The program was tested using `openmpi-3.1.1`.

The specific versions that we used in our runs are:

```
pyyaml==3.13
numpy==1.14.5
mpi4py==3.0.0
tensorflow==1.9.0
```

All of our runs were executed in a multi-computer environment, with NVIDIA K80 GPUs and Power8 processors running   
Linux (Red Hat Enterprise Linux 7.4 3).


---
### Running Q-NAS

The entire process is divided in 3 steps (1 python script for each):
1. Dataset preparation
2. Run architecture search
3. Retrain final architecture

Optionally, the user can run the script `run_profiling.py` to get the number of parameters and FLOPs   
of one of the discovered architectures.


#### 1. Dataset Preparation

The user can choose to work with one of these datasets: CIFAR-10 or CIFAR-100 (dataset details [here](https://www.cs.toronto.edu/~kriz/cifar.html)).

The script `run_dataset_prep.py` prepares the dataset, as described in the papers, to use for the    
architecture search and/or retraining. If the original CIFAR is already downloaded, just point to the folder   
with the files using the parameter `--data_path`. Otherwise, the script will download it for you and save it   
in the location defined by `--data_path`.

Here's an example of how to prepare the CIFAR-10 dataset limited to 10k examples for training and validation:

```shell script
python run_dataset_prep.py \
    --data_path cifar10 \
    --output_folder cifar_tfr_10000 \
    --num_classes 10 \
    --limit_data 10000
```

At the end of the process, the folder `cifar10/cifar_tfr_10000` has the following files:
>cifar_train_mean.npz  
data_info.txt  
test_1.tfrecords  
train_1.tfrecords  
valid_1.tfrecords  

The tfrecords files contains the images and labels, `data_info.txt` includes basic information about this dataset,   
and `cifar_train_mean.npz` is the numpy array with the mean of the training images.

This example shows how to prepare the CIFAR-100 dataset, with all the available training examples:

```shell script
python run_dataset_prep.py \
    --data_path cifar100 \
    --output_folder cifar_tfr \
    --num_classes 100 \
    --label_mode fine \
    --limit_data 0
```

Run `python run_dataset_prep.py --help` for additional parameter details.


#### 2. Run architecture search

All the configurable parameters to run the architecture search with Q-NAS are set in a _yaml configuration file_.   
This file sets 2 groups of parameters, namely: `QNAS` (parameters related to the evolution itself) and `train`   
(parameters related to the training session conducted to evaluate the architectures). The following template shows   
the type and meaning of each parameter in the configuration file: 

```yaml
QNAS:
    crossover_rate:      (float) crossover rate [0.0, 1.0]
    max_generations:     (int) maximum number of generations to run the algorithm
    max_num_nodes:       (int) maximum number of nodes in the network
    num_quantum_ind:     (int) number of quantum individuals
    penalize_number:     (int) maximum number of reducing layers in the networks without penalization
    repetition:          (int) number of classical individuals each quantum individual will generate
    replace_method:      (str) selection mechanism; 'best' or 'elitism'
    update_quantum_gen:  (int) generation frequency to update quantum genes
    update_quantum_rate: (float) rate for the quantum update (similar to crossover rate)
    save_data_freq:      (int) generation frequency to save train data of best model of current 
                           generation in csv files; if user do not want this, set it to 0. 

    params_ranges:       # Set to *value* if user wants this value instead of evolving the parameter
        decay:           (float) or (list); ignored if optimizer = Momentum
        learning_rate:   (float) or (list);
        momentum:        (float) or (list);
        weight_decay:    (float) or (list);

    function_dict: {'function_name': {'function': 'function_class', 
                                      'params': {'parameter': value}, 
                                      'prob': probability_value}}

train:
    batch_size:          (int) number of examples in a batch to train the networks.
    eval_batch_size:     (int) batch size for evaluation; if = 0, the number of valid image is used
    max_epochs:          (int) maximum number of epochs to train the networks.
    epochs_to_eval:      (int) fitness is defined as the maximum accuracy in the last *epochs_to_eval*
    optimizer:           (str) RMSProp or Momentum

    # Dataset
    dataset:             (str) Cifar10 or CIFAR100
    data_augmentation:   (bool) True if data augmentation methods should be applied
    subtract_mean:       (bool) True if the dataset mean image should be subtracted from images

    # Tensorflow
    save_checkpoints_epochs: (int) number of epochs to save a new checkpoint.
    save_summary_epochs:     (float) number of epochs (or fraction of an epoch) to save new summary
    threads:                 (int) number of threads for Tensorflow ops (0 -> number of logical cores)
```

We provide 3 configuration file examples in the folder `config_files`; one can use them as-is, or modify as needed.   
In summary, the files are:
- `config1.txt` evolves both the architecture and some hyperparameters of the network
- `config2.txt` evolves only the architecture and adopts penalization
- `config3.txt` evolves only the architecture with residual blocks and adopts penalization


This is an example of how to run architecture search for dataset `cifar10/cifar_tfr_10000` with `config1.txt`:

```shell script
mpirun -n 20 python run_evolution.py \
    --experiment_path my_exp_config1 \
    --config_file config_files/config1.txt \
    --data_path cifar10/cifar_tfr_10000 \
    --log_level INFO
```

The number of workers in the MPI execution must be equal to the number of classical individuals. In `config1.txt`,   
this number is 20 (_num_quantum_ind_ (=5) x _repetition_ (=4) = 20). The output folder `my_exp_config1` looks like this:

>12_7   
csv_data   
data_QNAS.pkl   
log_params_evolution.txt   
log_QNAS.txt

The folder `12_7` has the Tensorflow files for the best network in the evolution; in this case, is the individual   
number `7` found in generation `12`. The folder `csv_data` has csv files with training information of the    
individuals (loss and accuracy for the best individuals in some generations). Both of these directories are not used   
in later steps, they are just information that one might want to inspect.

The file `data_QNAS.pkl` keeps all the evolution data (chromosomes, fitness values, number of evaluations, best    
individual ID ...). All the parameters (configuration file and command-line) are saved in `log_params_evolution.txt`,   
and `log_QNAS.txt` logs the evolution progression.

It is also possible to continue a finished evolution process. Note that all the parameters will be set as in   
`log_params_evolution.txt`, ignoring the values in the file indicated by `--config_file`. The only parameter that can   
be overwritten is `max_generations`, so that one can set for how many generations the evolution will continue.   
To continue the above experiment for another 100 generations, the user can run:

```shell script
mpirun -n 20 python run_evolution.py \
    --experiment_path my_exp_config1/continue \
    --config_file config_files/config1.txt \
    --data_path cifar10/cifar_tfr_10000 \
    --log_level INFO \
    --continue_path my_exp_config1
```

Run `python run_evolution.py --help` for additional parameter details.


#### 3. Retrain network

After the evolution is complete, the final network can be retrained on the entire dataset (see papers for details).  
Here's an example of how to retrain the best network of the experiment saved in `my_exp_config1` for 300 epochs with 
the dataset in `cifar10/cifar_tfr`, using the scheme (optimizer and hyperparameters) of the evolution:

```shell script
python run_retrain.py \
    --experiment_path my_exp_config1 \
    --data_path cifar10/cifar_tfr \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 256 \
    --eval_batch_size 1000 \
    --threads 8 \
    --run_train_eval
```

After the training is complete, the directory `my_exp_config1/retrain` will contain the following files:

>best  
eval  
eval_test  
eval_train_eval  
checkpoint  
events.out.tfevents  
graph.pbtxt  
log_params_retrain.txt  
model.ckpt-52500.data-00000-of-00001  
model.ckpt-52500.index  
model.ckpt-52500.meta  

In the folder `best`, we have the best validation model saved. The file `log_params_retrain.txt` summarizes the 
training parameters. The other files and folders are generated by Tensorflow, including the last model saved, the   
graph and events for Tensorboard.


It is also possible to retrain the network with training schemes defined in the literature (check the help message for   
the `--lr_schedule` parameter). For example, to retrain the best network of experiment `my_exp_config2` using the   
`cosine` scheme, one can run:

```shell script
python run_retrain.py \
    --experiment_path my_exp_config2 \
    --data_path cifar10/cifar_tfr \
    --log_level INFO \
    --batch_size 256 \
    --eval_batch_size 1000 \
    --retrain_folder train_cosine \
    --threads 8 \
    --lr_schedule cosine \
    --run_train_eval
```

The script `run_retrain.py` also supports retraining any individual saved in `data_QNAS.pkl`: use the parameters   
`--generation` and `--individual` to indicate which one you want to train. Run `python run_retrain.py --help` for   
additional parameter details.


#### Profile architecture

If one wants to get the number of weights and the MFLOPs in a specific individual he/she can run `run_profiling.py`.   
For example, to get the values for individual `1` of generation `50` of the experiment saved in `my_exp_config3`, run:

```shell script
python run_profiling.py \
    --exp_path my_exp_config3 \
    --generation 50 \
    --individual 1
```

---
### Acknowledgements

This code was developed by Daniela Szwarcman while she was a PhD candidate in Electrical Engineering at PUC-Rio<sup>[1](#myfootnote1)</sup>   
and a PhD intern at IBM Research.

<a name="myfootnote1">1</a>. Advisor: Prof. Marley Vellasco ([PUC-Rio](https://www.puc-rio.br/index.html))  
    Co-advisor: Daniel Civitarese ([IBM Research](https://www.research.ibm.com/labs/brazil/))

