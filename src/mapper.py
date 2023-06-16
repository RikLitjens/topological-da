import kmapper as km
import pandas as pd
import numpy as np

df = pd.read_pickle('src/geolifePython7.pkl')
df = df.loc[df['user'].isin(np.arange(1,10))]
labels = df['label']
users = df['user']

df.drop(['time', 'label'], axis=1, inplace=True)

print(df.columns)

np_df = df.to_numpy()


mapper = km.KeplerMapper(verbose=1)

projected_data = mapper.fit_transform(np_df, projection=[1,2])

cover = km.Cover(n_cubes=10)

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, np_df, cover=cover)

# Visualize it
_ = mapper.visualize(graph, path_html="output/make_circles_keplermapper_output.html",
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
