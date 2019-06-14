# TrafficPredict
Pytorch implementation for the paper: [TrafficPredict: Trajectory Prediction for Heterogeneous Traffic-Agents](https://arxiv.org/abs/1811.02146) (AAAI), Oral, 2019

The repo has been forked initially from [Anirudh Vemula](https://github.com/vvanirudh)'s repository for his paper [Social Attention: Modeling Attention in Human Crowds](https://www.ri.cmu.edu/wp-content/uploads/2018/08/main.pdf) (ICRA 2018). If you find this code useful in your research then please also cite Anirudh Vemula's paper.

## Comparison of results:
|   Methods  | Paper  ADE | This repo ADE | Paper  FDE | This repo FDE |
|:----------:|:----------:|:-------------:|:----------:|:-------------:|
| pedestrian |    0.091   |     0.088     |    0.150   |     0.132     |
|   bicycle  |    0.083   |     0.075     |    0.139   |     0.115     |
|   vehicle  |    0.080   |     0.090     |    0.131   |     0.153     |
|    total   |    0.085   |     0.084     |    0.141   |     0.133     |

## Requirements

* Python 3
* Seaborn (https://seaborn.pydata.org/)
* PyTorch (http://pytorch.org/)
* Numpy
* Matplotlib
* Scipy

## How to Run
* First 'cd srnn'
* To train the model run python train.py (See the code to understand all the arguments that can be given to the command)
* To test the model run python sample.py --epoch=n where n is the epoch at which you want to load the saved model. (See the code to understand all the arguments that can be given to the command)
