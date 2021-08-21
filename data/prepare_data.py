import pandas as pd
import numpy as np
import geopandas as gpd
import os

create_geo_files = False
create_train_data = True
seq_len = 20
base_path = "../datasets/Ushant-Traffic"
texts_path = "../datasets/Ushant-Traffic/text_data"
jsons_path = "../datasets/Ushant-Traffic/json_data"
trajs = os.listdir(texts_path)
trajs_paths = [os.path.join(texts_path, traj_name) for traj_name in trajs]

if create_geo_files:
    for path in trajs_paths:
        df = pd.read_csv(path, delimiter = ";")
        gdf = gpd.GeoDataFrame(df, geometry=(gpd.points_from_xy(df.x, df.y)))
        file_name = os.path.splitext(os.path.basename(path))[0] + ".json"
        gdf.to_file(os.path.join(jsons_path, file_name), driver="GeoJSON")

if create_train_data:
    train_folder = os.path.join(base_path, 'train')
    valid_folder = os.path.join(base_path, 'valid')
    test_folder = os.path.join(base_path, 'test')

    all_data = []
    train_list = []
    test_list = []
    valid_list = []

    id = 1
    for path in trajs_paths:
        df = pd.read_csv(path, delimiter=";")
        df['id'] = id
        df_length = df.shape[0]//seq_len
        for idx in range(df_length):
            all_data.append(df.values[idx:idx+seq_len])
        id += 1
pass