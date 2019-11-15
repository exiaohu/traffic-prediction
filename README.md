# General Traffic Prediction by Graph Neural Networks

This is PyTorch implementations of several papers.

## Requirements
- scikit-learn
- PyYAML
- tqdm
- networkx
- numpy
- tensorboardX
- torch



Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data Preparation
Same as [DCRNN](https://github.com/liyaguang/DCRNN), the traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY), i.e., `metr-la.h5` and `pems-bay.h5`, are available at [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g), and should be
put into the `data/` folder.
The `*.h5` files store the data in `panads.DataFrame` using the `HDF5` file format. Here is an example:

|                     | sensor_0 | sensor_1 | sensor_2 | sensor_n |
|:-------------------:|:--------:|:--------:|:--------:|:--------:|
| 2018/01/01 00:00:00 |   60.0   |   65.0   |   70.0   |    ...   |
| 2018/01/01 00:05:00 |   61.0   |   64.0   |   65.0   |    ...   |
| 2018/01/01 00:10:00 |   63.0   |   65.0   |   60.0   |    ...   |
|         ...         |    ...   |    ...   |    ...   |    ...   |


Here is an article about [Using HDF5 with Python](https://medium.com/@jerilkuriakose/using-hdf5-with-python-6c5242d08773).

Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.
```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Graph Construction
 As the currently implementation is based on pre-calculated road network distances between sensors, it currently only
 supports sensor ids in Los Angeles (see `data/sensor_graph/sensor_info_201206.csv`).
```bash
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl
```
Besides, the locations of sensors in Los Angeles, i.e., METR-LA, are available at [data/sensor_graph/graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv).