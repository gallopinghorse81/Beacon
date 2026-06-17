

## Introduction

This repository contains the model implementation of Beacon, the preprocessing pipeline, BERTopic-based aspect generation, and the main training and evaluation scripts for our paper.

## Overview

The main files are:

- pro_data/data_pro_practice_stranger_final.py: data preprocessing script.
- pro_data/data_pro_for_bertopic_final.py: aspect-vector generation script based on BERTopic.
- main_context.py: main training and testing script.
- models/StrangerContext.py: implementation of the model.
- config_my_own/config_my_own.py: dataset paths and hyperparameter configuration.

## Environment Requirement

Python=3.11.7
Pytorch=2.1.1
Bertopic=0.16.4

A CUDA-enabled GPU is recommended because the preprocessing and training scripts adopt GPU tensors.

## Download Data

Our paper uses six datasets, including datasets of Auto, Video, Tools, Baby, Yelp2013, and Yelp2017. Put the downloaded raw data files into the folder "pro_data/". The Amazon review datasets can be downloaded from [Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html), and the Yelp datasets can be downloaded from [Yelp Open Dataset](https://www.yelp.com/dataset).


## Pretrained Resources

Put the required pretrained resources into the corresponding local folders by downloading from [Hugging Face](https://huggingface.co/) and [GoogleNews-vectors-negative300](https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300/data).

The downloaded pretrained resources should include:

- pro_data/bert-base-uncased/
- pro_data/GoogleNews-vectors-negative300.bin
- dataset/all-mpnet-base-v2/

## Data Preprocessing

Run the preprocessing script:

```bash
cd pro_data
python data_pro_practice_stranger_final.py
```

## Generate BERTopic-based Aspect Vectors

After preprocessing, run:

```bash
cd pro_data
python data_pro_for_bertopic_final.py
```

This script generates BERTopic aspect vectors for 1, 3, 5, and 7 aspects. 

## Training

Train the model by:

```bash
python main_context.py
```

The default dataset is Video, pecific parameters such as dataset, batch size, learning rate, GPU id, and aspect number can be modified in main_context.py and config_my_own/config_my_own.py.

Training logs are saved in:

```text
log/main_context.log
```

Model checkpoints are saved in:

```text
checkpoints/
```

## Acknowledgement

The structure of this repository is based on the [Neu-Review-Rec](https://github.com/ShomyLiu/Neu-Review-Rec) implementation of NRPA [1], and we thank the authors for releasing their implementation.

## References

[1] H. Liu, F. Wu, W. Wang, X. Wang, P. Jiao, C. Wu, and X. Xie, "[NRPA: Neural Recommendation with Personalized Attention](https://arxiv.org/abs/1905.12480)," arXiv:1905.12480, 2019.
