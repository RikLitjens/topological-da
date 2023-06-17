import open3d as o3d
import helpers
import configparser

config = configparser.RawConfigParser()
config.read(r"Cherry-trees\config\config.ini")
local_path = config.get('DATA', 'PATH')

print(local_path)