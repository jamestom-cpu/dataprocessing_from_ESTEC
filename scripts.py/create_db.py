from tqdm import tqdm
import numpy as np
import h5py
import os, sys
import re

sys.path.append(r"C:\Users\tomas\Desktop\phd\dataprocessing")

from my_packages.my_hdf5 import *
from my_packages.directory_data import make_all_generators

def get_numpy_from_batch(batch):
    readings = [list(read_csv(csv_file_path)) for csv_file_path in batch]

    # I must transpose so that the shape has the number of points in the last position and 2 - ie time and amplitude - in 
    # second to last place
    
    return np.asarray(readings).transpose(0,2,1)

def read_csv(filename):
    for row in open(filename):
        yield list(map(lambda x: float(x),row.split(",")[-2:]))




# Function to extract coordinates
def extract_coordinates(filename):
    # Regular expression pattern for x and y coordinates
    pattern_coords = r'x([0-9]+\.[0-9]{2})y([0-9]+\.[0-9]{2})'

    match = re.search(pattern_coords, filename)
    if match:
        x = float(match.group(1))
        y = float(match.group(2))
        return [x, y]
    else:
        return None

def parse_batch(batch):
    all_coords = np.asarray([extract_coordinates(filename) for filename in batch])
    return np.unique(all_coords, axis=0)
    

def measurements2dataset(library, batch_generator, group_name, compression_level=4, max_n_batches=None):
    number_of_batches = len(batch_generator)
    if max_n_batches is None:
        max_n_batches = number_of_batches+1
    datapoints = int(100e3)
    batch_size = batch_generator.batch_size

    dataset_shape = (batch_size, number_of_batches, 2, datapoints)

    group_attr = dict(
        observation_interval = "10us",
        number_of_points = datapoints,
        max_frequency_resolution = 1/datapoints, 
        structure_of_numpy_array = ("batch_size", "number_of_batches", "time, amplitude", "datapoints") 
    )

    if not exists(library):
        build_hdf5(name=library)

    with h5py.File(library, "a") as f:
        # create group 
        group = f.require_group(group_name)
        group.attrs.update(group_attr)

        # create dataset for measurements
        measurement_dataset = group.require_dataset(
            "measurements", 
            shape=dataset_shape, 
            dtype="float64", 
            chunks=(batch_size, 1, 2, datapoints), 
            compression=compression_level
        )

        # create dataset for coordinates
        coordinates_dataset = group.require_dataset(
            "coordinates", 
            shape=(number_of_batches, 2), 
            dtype="float64", 
            compression=compression_level
        )

        

        for batch_number, batch in tqdm(enumerate(batch_generator), total=number_of_batches):
            if batch_number >= max_n_batches:
                break
            parsed_batch = parse_batch(batch)
            assert len(parsed_batch) == 1, "batch contains more than one coordinate"
            
            try:
                np_batch = get_numpy_from_batch(batch)
                # Adjust the batch if its size is not 50
                if np_batch.shape[0] < 50:
                    padding = np.full((50 - np_batch.shape[0],) + np_batch.shape[1:], np.nan)
                    np_batch = np.concatenate((np_batch, padding))
                elif np_batch.shape[0] > 50:
                    np_batch = np_batch[:50]
                    
                # update the datasets with the batches - this way we can save the data even if the batch fails to load
                measurement_dataset[:, batch_generator.number_of_batches_done-1, :, :] = np_batch
                coordinates_dataset[batch_generator.number_of_batches_done-1, :] = np.asarray(parsed_batch[0])
            except Exception as e:
                print(f"Batch failed to load with error: {str(e)}")
                continue    
        return None
            
filename = "measurements.h5"
generators = make_all_generators(filename, return_fullpaths=True)

filepath = r"E:\measurement_data.h5"

key, gen = list(generators.items())[1]
print("key:", key)
print("gen:", gen)

measurements2dataset(filepath, gen, key, compression_level=5)