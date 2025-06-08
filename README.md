# PM2.5-GNN

PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting

## Dataset

- Download dataset **KnowAir** from [Google Drive](https://drive.google.com/open?id=1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L) or [Baiduyun](https://pan.baidu.com/s/18D6Etl5Lm1E4vOLVrX0ZAw) with code `t82d`.

## KnowAir-V2

🚀 Dataset Update: Announcing KnowAir-V2! 🚀

We are excited to announce a major upgrade to the original KnowAir (PM2.5-GNN) dataset with the official release of KnowAir-V2! This is a brand-new, higher-quality benchmark dataset for air quality forecasting.

Key improvements in KnowAir-V2 include:
- Longer Temporal Span: Data covers from 2016 to 2023.
- Richer Variables: Includes not only PM2.5 but also O3 and more related meteorological variables.
- Higher Data Quality: The data has undergone rigorous preprocessing and imputation, reaching an operational-level standard.

For all new research and projects, we strongly recommend using KnowAir-V2. This dataset is designed to provide a powerful benchmarking platform for more advanced spatio-temporal prediction models that integrate physical-chemical knowledge, such as PCDCNet.

How to Access and Cite
Dataset Download (KnowAir-V2):

- Wang, S., Cheng, Y., Meng, Q., Saukh, O., Zhang, J., Fan, J., Zhang, Y., Yuan, X., & Thiele, L. (2025). KnowAir-V2: A Benchmark Dataset for Air Quality Forecasting with PCDCNet [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15614907

- Related Paper (PCDCNet):
Please refer to the paper: "PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints" (arXiv:2505.19842). https://www.arxiv.org/abs/2505.19842

## Requirements

```
Python 3.7.3
PyTorch 1.7.0
PyG: https://github.com/rusty1s/pytorch_geometric#pytorch-170
```

```bash
pip install -r requirements.txt
```

## Experiment Setup

open `config.yaml`, do the following setups.

- set data path after your server name. Like mine.

![](https://tva1.sinaimg.cn/large/0081Kckwly1gjy8kojsfmj30i202g746.jpg)

```python
filepath:
  GPU-Server:
    knowair_fp: /data/wangshuo/haze/pm25gnn/KnowAir.npy
    results_dir: /data/wangshuo/haze/pm25gnn/results

```

- Uncomment the model you want to run.

```python
#  model: MLP
#  model: LSTM
#  model: GRU
#  model: GC_LSTM
#  model: nodesFC_GRU
   model: PM25_GNN
#  model: PM25_GNN_nosub
```

- Choose the sub-datast number in [1,2,3].

```python
 dataset_num: 3
```

- Set weather variables you wish to use. Following is the default setting in the paper. You can uncomment specific variables. Variables in dataset **KnowAir** is defined in `metero_var`.

```python
  metero_use: ['2m_temperature',
               'boundary_layer_height',
               'k_index',
               'relative_humidity+950',
               'surface_pressure',
               'total_precipitation',
               'u_component_of_wind+950',
               'v_component_of_wind+950',]

```

## Run

```bash
python train.py
```

## Reference

Paper: https://dl.acm.org/doi/10.1145/3397536.3422208

```
@inproceedings{10.1145/3397536.3422208,
author = {Wang, Shuo and Li, Yanran and Zhang, Jiang and Meng, Qingye and Meng, Lingwei and Gao, Fei},
title = {PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting},
year = {2020},
isbn = {9781450380195},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3397536.3422208},
doi = {10.1145/3397536.3422208},
abstract = {When predicting PM2.5 concentrations, it is necessary to consider complex information sources since the concentrations are influenced by various factors within a long period. In this paper, we identify a set of critical domain knowledge for PM2.5 forecasting and develop a novel graph based model, PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world dataset, we validate the effectiveness of the proposed model and examine its abilities of capturing both fine-grained and long-term influences in PM2.5 process. The proposed PM2.5-GNN has also been deployed online to provide free forecasting service.},
booktitle = {Proceedings of the 28th International Conference on Advances in Geographic Information Systems},
pages = {163–166},
numpages = {4},
keywords = {air quality prediction, graph neural network, spatio-temporal prediction},
location = {Seattle, WA, USA},
series = {SIGSPATIAL '20}
}
```
