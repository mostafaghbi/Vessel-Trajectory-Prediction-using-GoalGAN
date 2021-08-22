import pandas as pd
import geopandas as gpd
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
np.random.seed(100)

create_geo_files = False
create_mask = False
create_train_data = True

seq_len = 20
base_path = "../datasets/Ushant-Traffic"
texts_path = "../datasets/Ushant-Traffic/text_data"
os.makedirs(texts_path, exist_ok=True)
jsons_path = "../datasets/Ushant-Traffic/json_data"
os.makedirs(jsons_path, exist_ok=True)
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
    os.makedirs(train_folder, exist_ok=True)
    valid_folder = os.path.join(base_path, 'valid')
    os.makedirs(valid_folder, exist_ok=True)
    test_folder = os.path.join(base_path, 'test')
    os.makedirs(test_folder, exist_ok=True)

    all_data = []

    id = 1
    for path in trajs_paths:
        df = pd.read_csv(path, delimiter=";")
        df = df.get(['x', 'y'])
        df_length = df.shape[0]//seq_len
        for idx in range(0, df_length*seq_len, seq_len):
            all_data.append(df.values[idx:idx+seq_len])

    # Split data to train, valid and test
    train_length = int(0.7 * len(all_data))
    test_length = int(0.2 * len(all_data))
    valid_length = len(all_data) - train_length - test_length
    np.random.shuffle(all_data)

    # Create train
    train_temp = np.array(all_data[0:train_length])
    train_data = np.zeros((train_temp.shape[0], 20, 4))
    train_data[:, :, -2] = train_temp[:, :, 0]
    train_data[:, :, -1] = train_temp[:, :, 1]
    train_data = train_data.reshape((train_temp.shape[0]*seq_len, 4))
    train_data[:, 0] = np.array(range(0, train_temp.shape[0]*seq_len, 1))
    train_data[:, 1] = np.repeat(np.array(range(0, train_temp.shape[0], 1)), 20)
    pd.DataFrame(train_data).to_csv(os.path.join(train_folder, "train.csv"), sep="\t", index=False)
    # pd.read_csv(os.path.join(train_folder, "train.csv"), delimiter="\t", index_col=False)

    # Create test
    test_temp = np.array(all_data[train_length: train_length+test_length])
    test_data = np.zeros((test_temp.shape[0], 20, 4))
    test_data[:, :, -2] = test_temp[:, :, 0]
    test_data[:, :, -1] = test_temp[:, :, 1]
    test_data = test_data.reshape((test_temp.shape[0]*seq_len, 4))
    test_data[:, 0] = np.array(range(0, test_temp.shape[0]*seq_len, 1))
    test_data[:, 1] = np.repeat(np.array(range(0, test_temp.shape[0], 1)), 20)
    pd.DataFrame(test_data).to_csv(os.path.join(test_folder, "test.csv"), sep="\t", index=False)

    # Create valid
    valid_temp = np.array(all_data[train_length+test_length:])
    valid_data = np.zeros((valid_temp.shape[0], 20, 4))
    valid_data[:, :, -2] = valid_temp[:, :, 0]
    valid_data[:, :, -1] = valid_temp[:, :, 1]
    valid_data = valid_data.reshape((valid_temp.shape[0]*seq_len, 4))
    valid_data[:, 0] = np.array(range(0, valid_temp.shape[0]*seq_len, 1))
    valid_data[:, 1] = np.repeat(np.array(range(0, valid_temp.shape[0], 1)), 20)
    pd.DataFrame(valid_data).to_csv(os.path.join(valid_folder, "valid.csv"), sep="\t", index=False)


if create_mask:
    im = gdal.Open(os.path.join(base_path, "scene.tif"))
    projection = im.GetProjection()
    geo_transform = im.GetGeoTransform()
    im_ = im.GetRasterBand(1)
    im_array = im_.ReadAsArray()
    rows, cols = im_array.shape
    mask = im_array.copy()
    mask[mask == 1] = 0
    mask[mask == 2] = 255

    # Copy geotiff data between two images
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(os.path.join(base_path, 'mask.tif'), cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(mask)
