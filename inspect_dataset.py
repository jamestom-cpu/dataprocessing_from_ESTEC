import h5py
from pprint import pprint
from my_packages.directory_data import make_all_generators, measurements2dataset



def inspect_dataset(library, group_name, dataset_name="measurements"):
    '''
    Prints the shape, dtype, and compression level of a specified dataset within a group.
    '''
    with h5py.File(library, 'r') as f:
        dataset = f[group_name][dataset_name]
        print(f'Dataset "{dataset_name}" within group "{group_name}":')
        print(f'Shape: {dataset.shape}')
        print(f'Dtype: {dataset.dtype}')
        print(f'Compression level: {dataset.compression}')

def inspect_group(library, group_name):
    '''
    Prints the names and attributes of all datasets within a specified group.
    '''
    with h5py.File(library, 'r') as f:
        group = f[group_name]
        print(f'Group "{group_name}":')
        for attr_name, attr_value in group.attrs.items():
                print(f'  {attr_name}: {attr_value}')
        for name, dataset in group.items():
            print(f'\nDataset "{name}":')
            print(f'Shape: {dataset.shape}')
            print('Attributes:')
            for attr_name, attr_value in dataset.attrs.items():
                print(f'  {attr_name}: {attr_value}')

def inspect_file(library):
    '''
    Prints the names of all groups and datasets in the file.
    '''
    with h5py.File(library, 'r') as f:
        print(f'File "{library}":')
        f.visit(print)


def get_data_from_hdf5(hdf5_path, group, index):
    with h5py.File(hdf5_path, 'r') as f:
        dataset = f[group]["measurements"]
        data = dataset[index]  # Get the data at the given index
    return data

def get_batch_from_hdf5(hdf5_path, group, index):
    complex_index = (slice(None), index, slice(None), slice(None))
    return get_data_from_hdf5(hdf5_path, group, complex_index)
    

hdf5_path = r"E:\measurement_data_1.h5"
group = 'XFE_04s'
  # Use a tuple of slices to get the first 100000 values at position 0,0,0

data = get_batch_from_hdf5(hdf5_path, group, 0)


filepath = r"E:\measurement_data_1.h5"

inspect_dataset(filepath, "XFE_04s")
inspect_group(filepath, "XFE_04s")



