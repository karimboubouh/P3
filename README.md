# On the Energy Efficiency of P2P Personalized Machine Learning using Mobile Devices

This repository contains the code to run simulations for the "*On the Advantages of P2P ML on Mobile Devices*" paper, submitted in *ACM e-Energy 2022* conference.
The repository includes the implementation of the P3 algorithm on a simulated environment, and the implementation of the P3 algorithm on Android.

### Requirements

| Package     | Version |
|-------------|---------|
| python      | 3.7     |
| pytorch     | 1.9.1   |
| torchvision | 0.10.1  |
| numpy       | 1.21.3  |

### Data

We use the MNIST dataset for energy analysis and model evaluation but implementation supports also CIFAR10. Datasets will be downloaded automatically in the `/data` folder with the first run of the algorithm. 

### ML Engine

We have implemented the code to run on two ML engines. `Pytorch` based ML models and `Numpy` based models to support ML training on android devices.
To configure the ML Engine, update the following line in `src/conf.py` 
``
ML_ENGINE = "PyTorch"  # "NumPy" or "PyTorch"
``
**NB:** Android implementation does not support PyTorch.

## Evaluation of P3 

### Configuration

To build the random topology of the P2P network, we use the graph density parameter `rho` to estimate the number of neighbors for each nodes. In our experiments, we use the following:

- For 10 peers: `rho=0.4`
- For 100 peers: `rho=0.8`
- For 300 peers: `rho=0.95`

The main algorithm parameters are the following:

| Argument     | Description                                                                       |
|--------------|-----------------------------------------------------------------------------------|
| --mp         | Use message passing (MP) via sockets or shared memory (SM)  (default: MP)         |
| --rounds     | Number of rounds of collaborative training (default: 500)                         |
| --num_users  | Number of peers joining the P2P network (default: 100)                            |
| --epochs     | Number of epochs for local training (default: 2)                                  |
| --batch_size | Batch size (default: 64)                                                          |
| --lr         | Learning rate (default: 0.1)                                                      |
| --momentum   | SGD momentum (default: 0.9)                                                       |
| --model      | ML model (default: mlp) Multi-Layer-Perceptron                                    |
| --dataset    | Dataset (default: mnist)                                                          |
| --iid        | Data distribution, default set to IID. Set to 0 for non-IID.                      |
| --unequal    | Whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits) |

### Execution of Personalized P2P (P3)

To reproduce the experiments of model performance in the paper use the following command:

- IID data partitioning

`python main.py --num_users=100 --model=mlp --dataset=mnist --iid=1 --round=500`

- Balanced non-IID partitioning

`python main.py --num_users=100 --model=mlp --dataset=mnist --iid=0 --unequal=0 --round=500`

- Unbalanced non-IID partitioning

`python main.py --num_users=100 --model=mlp --dataset=mnist --iid=0 --unequal=1 --round=500`

- Network density:

Change the values of parameter $\rho$

```python
topology = random_graph(models, rho=0.95)  # 0, 0.4, 0.7, 0.95, 0.99
```



### Execution of Centralized Learning (CL)

`python main_CL.py--model=mlp --dataset=mnist`

### Execution of Model Propagation (MP)

`python main_MP.py --num_users=100 --model=mlp --dataset=mnist --iid=1 --round=500`

### Execution of FedAvg (FL)

`python main_FL.py --num_users=100 --model=mlp --dataset=mnist --iid=1 --round=500`

## Energy Analysis

To perform energy analysis of P3 on the Linux server (Ubuntu 20.04), we developed two methods of energy readings:

- Evaluating the whole program by running the `run.sh` script.
- Evaluating a given method of the algorithm using python decorators.

**NB:** you need to disable virtualization from the bios as we shield the program to one physical core.

### Requirement

We have used the following packages: `powerstat`, `cset-shield`, `cpupower`.

### Energy consumption of the whole program

To measure the energy consumption of the whole program run the following:

`./run.sh -c 0 -p avg -r 1 -d 2 -e "python main.py" -a "--num_users=100 --model=mlp"`

Run `./run.sh -h` to get a list of the available options and what are used for.

### Energy consumption of a method

To measure the energy consumption of a given method, use the `@measure_energy` decorator.

For example to evaluation the energy consumption of the local learning step, add the following: 

````python
@measure_energy
def local_training(self, device='cpu', inference=True):
	log('event', 'Starting local training ...')
	...
````

End.