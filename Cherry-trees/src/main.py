import open3d as o3d
import helpers
import configparser

import os

this_folder = os.path.dirname(os.path.abspath(__file__))
init_file = os.path.join(this_folder, 'config.init')
print(init_file)

config = configparser.RawConfigParser()
config.read(init_file)
local_path = config.get('DATA', 'PATH')

print(local_path)

# bagnumber + image_number