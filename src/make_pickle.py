import pandas as pd
import numpy as np
import read_geolife
from IPython.display import Markdown

# SLOW!

df = read_geolife.read_all_users('Geolife\Geolife\Data')
df.to_pickle('geolifePython7.pkl')