import igraph as ig
import numpy as np
from haversine import distance
import networkx as nx
import matplotlib.pyplot as plt

graph = ig.read('FatMan.graphml')

latlon=[] 
lats=list(graph.vs['Latitude']) #read the latitute attribute
lons=list(graph.vs['Longitude']) #read the longitude attribute
lat=np.array([lats])
lon=np.array([lons])
latlon= np.concatenate((lat.T,lon.T),axis=1)
lat_lon1=latlon.tolist()

edges = graph.get_edgelist()
link_delays = []
for edge in edges :
  delay = distance([(lat_lon1[edge[0]][0], lat_lon1[edge[0]][1]),(lat_lon1[edge[1]][0], lat_lon1[edge[1]][1])])
  link_delays.append(delay)

nodes = [ int(id) for id in graph.vs['id'] ]


graphNx = nx.Graph()
graphNx.add_nodes_from(nodes)
graphNx.add_edges_from(edges)

index = 0
for edge in edges:
  graphNx.edges[edge[0], edge[1]]['weight'] = link_delays[index]
  index += 1

subax1 = plt.subplot(121)
nx.draw(graphNx, with_labels=True, font_weight='bold')
plt.show()



shortestDistanceMatrix = [ [ -1 for j in range(len(nodes))] for j in range(len(nodes))]
for i in range(len(nodes)):
  for j in range(len(nodes)):
    if i==j:
      shortestDistanceMatrix[i][j] = 0
    else:
      total_distance = 0
      shortest_path = nx.shortest_path(graphNx, source= i, target= j)
      index = 0
      while index < len(shortest_path) - 1:
        total_distance += graphNx.edges[shortest_path[index], shortest_path[index+1]]['weight']
        index += 1
      shortestDistanceMatrix[i][j] = total_distance

print(shortestDistanceMatrix)