import numpy as np
import os
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads
from shapely.geometry import Point
import matplotlib.pyplot as plt
from rasterstats import point_query

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)
# Get the directory name of the current file
current_folder = os.path.dirname(current_file_path)

cityname = "eindhoven"
query_name = "eindhoven"
saving_name = "WholeEHV"
selected_districts = [] # ["Binnenstad", "Witte Dame", "Bergen"]  # If you want to select a specific district, write the name here
filepath = current_folder + cityname + ".graphml"

# Define the place and tags for supermarkets
tags = {"amenity": ['school','parking_entrance','place_of_worship', 'fuel', 'pharmacy', 'parcel_locker', 'kindergarten', 
                    'car_wash', 'library', 'grave_yard', 'bank', 'post_office', 'townhall', 'parking_exit', 'locker', 'supermarket'],
        "shop": ["supermarket", "mall", "department_store"]
        }

### Load the neighborhoods data: Name, population and shape

nbh_shapes = gpd.read_file(current_folder+"/EindhovenOpenData/GeoNeighborhoods.geojson").rename(columns={"buurtnaam": "District"})
nbh_population = pd.read_csv(current_folder+"/EindhovenOpenData/PopNeighborhoods.csv", sep=";", names=["Name", "Population"])

if set(nbh_shapes["District"]) ^ set(nbh_population['Name']) != set():
    print("Neighborhoods names do not match")
    print(set(nbh_shapes["Name"]) ^ set(nbh_population['Name']))
else:
    district_names = list(nbh_shapes["District"])

nbh_population['Population'] = nbh_population['Population'].replace('x', 0).astype(int)

nbh_shapes = nbh_shapes.merge(nbh_population, left_on="District", right_on="Name")
print("Merged GeoDataFrame:")

desired_columns = ["District", "Population", "geometry"]
neighbourhoods_df = gpd.GeoDataFrame(nbh_shapes[desired_columns])

### Load or download the city OpenStreetMaps network

if saving_name == "":
    print("Loading saved map")
    referenced_path = ""
    graph_files = os.listdir(current_folder)
    for filename in graph_files:
        if cityname + ".graphml" in filename:
            referenced_path = current_folder +'/'+ filename
            print("Found city name in repository, uploading")
    interest_points_gdf = pd.read_csv(current_folder+'/'+cityname+"_interest_points.csv")
    interest_points_gdf['geometry'] = interest_points_gdf['geometry'].apply(loads)
    if len(selected_districts) >= 1:
        if set(selected_districts).issubset(set(neighbourhoods_df['District'].unique())): 
            print("Districts found")
            neighbourhoods_df = neighbourhoods_df[neighbourhoods_df['District'].isin(selected_districts)]
        else:
            raise "District not found or mentioned"
    interest_points_gdf = gpd.GeoDataFrame(interest_points_gdf, geometry='geometry', crs="EPSG:4326")
    G = ox.load_graphml(referenced_path)
    print("Saved map loaded")
else:
    print("Querying map")
    savepath = current_folder + saving_name + ".graphml"
    interest_points_gdf = ox.features_from_place(query_name, tags)
    interest_points_gdf = interest_points_gdf[['amenity', 'name', 'geometry']]
    interest_points_gdf.to_csv(current_folder+cityname+"_interest_points.csv", columns=["amenity", "name", "geometry"])
    if len(selected_districts) >= 1:
        if set(selected_districts).issubset(set(neighbourhoods_df['District'].unique())): 
            print("Districts found")
            neighbourhoods_df = neighbourhoods_df[neighbourhoods_df['District'].isin(selected_districts)]
        else:
            raise "District not found or mentioned"
    G = ox.graph_from_place(query_name, network_type="walk")
    ox.save_graphml(G, savepath)
    print("Maps queried and saved")

# print("Plotting neighbourhoods shapes")
# import matplotlib.pyplot as plt
# ax = neighbourhoods_df.boundary.plot()
# ax.set_axis_off()
# plt.show()  

interest_points_coords = interest_points_gdf[['geometry']].apply(lambda x: x.geometry.centroid.coords[0], axis=1)
print("interest points coordinates extracted")

# Perform spatial join
joined_gdf = gpd.sjoin(interest_points_gdf, neighbourhoods_df, how="left", predicate="within")

# Add the area name to interest_points_gdf
interest_points_gdf['District'] = joined_gdf['District']
# print(interest_points_gdf.head())

# Find the nearest node in the graph for each amenity
interest_points_gdf['nearest_node'] = interest_points_coords.apply(lambda coord: ox.distance.nearest_nodes(G, coord[0], coord[1]))
# distances_between_interest_point_and_closest_intersection = interest_points_coords.apply(lambda coord: ox.distance.nearest_nodes(G, coord[0], coord[1], return_dist=True)[1])
# print(f"Min distance is {distances_between_interest_point_and_closest_intersection.min()}\nMax distance is {distances_between_interest_point_and_closest_intersection.max()}\n(95,99,99.9) percentiles are {distances_between_interest_point_and_closest_intersection.quantile([0.95,0.99,0.999])}")
print("Nearest nodes found")

### Add interest_points as attributes to the corresponding nodes
for idx, row in interest_points_gdf.iterrows():
    node = row['nearest_node']
    if 'locker_possible' not in G.nodes[node]:
        G.nodes[node]['locker_possible'] = 'locker'

# Create a GeoDataFrame from the node positions
geometry = list(map(lambda node_info: Point(node_info[1]['x'], node_info[1]['y']), G.nodes(data=True)))
nodes_gdf = gpd.GeoDataFrame(zip(list(G.nodes), geometry), columns=['node','geometry'], crs="EPSG:4326")

# Perform spatial join to find which district each node is in
# Assume that districts are a partition of Eindhoven, no overlaps
joined_gdf = gpd.sjoin(nodes_gdf, neighbourhoods_df, how="left", predicate="within")
# if selected_district in joined_gdf['District'].unique():
#     joined_gdf = joined_gdf[joined_gdf['District'] == selected_district]

# Add district information to the nodes in G
for idx, row in joined_gdf.iterrows():
    G.nodes[row['node']]['District'] = row['District']

# Calculate the population per node for each district
district_population = neighbourhoods_df.set_index('District')['Population']  # Replace 'district_column' and 'population' with actual column names
district_node_counts = joined_gdf['District'].value_counts()
population_per_node = {district : district_population[district] / district_node_counts[district] for district in district_node_counts.index}	

no_population_districts = {district for district in district_node_counts.index if district_population[district] == 0.0}
nodes_to_be_removed_for_no_population = set()
for node, data in G.nodes(data=True):
    if data.get('District') in no_population_districts:
        nodes_to_be_removed_for_no_population.add(node)
G.remove_nodes_from(nodes_to_be_removed_for_no_population)

if len(selected_districts)>1 and set(selected_districts).issubset(set(neighbourhoods_df['District'].unique())): 
    print("Cancel call")
    nodes_to_be_removed = set()
    for node, data in G.nodes(data=True):
        district = data.get('District')
        if district in selected_districts:
            G.nodes[node]['node_population'] = population_per_node[district]
        else:
            nodes_to_be_removed.add(node)
    G.remove_nodes_from(nodes_to_be_removed)
else:
    print("Keep call")
    for node, data in G.nodes(data=True):
        district = data.get('District')
        G.nodes[node]['node_population'] = population_per_node[district]

### Optionally, save the updated graph
if selected_districts:
    ox.save_graphml(G, current_folder + '/' +  cityname + "_with_districts_"+"_".join(selected_districts)+".graphml")
else:
    ox.save_graphml(G, current_folder + '/' +  cityname + "_with_possible_locker_locations.graphml")
print("Graph saved with possible locker locations")

# # Identify nodes with the attribute 'locker_possible'
# locker_nodes = [node for node, data in G.nodes(data=True) if data.get('locker_possible') == 'possible_locker']
# Create a color map for the nodes
node_colors = ['red' if data.get('locker_possible') == 'locker' else 'white' for _, data in G.nodes(data=True)]
node_sizes = [50 if data.get('locker_possible') == 'locker' else 10 for _, data in G.nodes(data=True)]
print("Color and size map created")

### Plot the graph
fig, ax = ox.plot_graph(G, node_color=node_colors, node_size=node_sizes, edge_color='gray', bgcolor='black', show=False, close=False)
plt.show()

print("Info about the graph:")
print("Number of nodes: ", len(G.nodes))
print("Number of edges: ", len(G.edges))
print("Number of interest points: ", len(interest_points_gdf))
# print("Degree: ", G.degree)


# # 2. Load the LandScan raster file (replace 'landscan_file.tif' with the actual file path)
# landscan_raster = "./landscan-global-2021-colorized.tif"

# # 3. Use `point_query` to extract population density for each node
# G.nodes["population_density"] = point_query(
#     G.nodes, 
#     landscan_raster, 
#     interpolate="nearest"  # Options: 'nearest', 'bilinear', etc.
# )

# nodes_with_lockers = [node for node, data in G.nodes(data=True) if data.get('locker_possible') == 'locker']
# print("Nodes with lockers: ", len(nodes_with_lockers))
# some_locker_node = nodes_with_lockers[2]