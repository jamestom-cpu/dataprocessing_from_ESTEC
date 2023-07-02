import numpy as np
from pprint import pprint
import os 

from my_packages.directory_data import make_all_generators, measurements2dataset

filename = "measurements.h5"
generators = make_all_generators(filename, return_fullpaths=True)

external_drive_path = "E:/"
archive_name = "full_measurement_data.h5"
archive_path = os.path.join(external_drive_path, archive_name)

for key, generator in generators.items():
    print("current key:", key)
    print("starting to save the data to the archive")
    measurements2dataset(archive_path, generator, key, compression_level=5)
    print("finished saving the data to the archive")
    print("moving to the next key")

print("finished saving all the data to the archive")
