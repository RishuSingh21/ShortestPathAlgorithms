#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys


def get_path(source, end, parent):
    short_path = [end]
    while source != end:
        short_path.append(parent[end])
        end = parent[end]

    short_path.reverse()
    return short_path


def min_dist(vertices, is_visited, dist):
    k=0
    min_d = float("inf")

    for v in vertices:
        if dist[v] < min_d and is_visited[v] == False:
            min_d = dist[v]
            k = v
    return k


def dijkstra(source, adjmatrix, vertices):
    dist = {}
    is_visited = {}

    for v in vertices:
        #initializing dist with infinity and is_visited with False
        dist[v] = float("inf")
        is_visited[v] = False

    dist[source] = 0
    parent = {source: -1}

    for m in vertices:

        near_vertex = min_dist(vertices, is_visited, dist)
        is_visited[near_vertex] = True

        for v in vertices:
            if adjmatrix[near_vertex][v] > 0 and adjmatrix[near_vertex][v] != float("inf") and                     is_visited[v] == False:
                if dist[v] > dist[near_vertex] + adjmatrix[near_vertex][v]:
                    dist[v] = dist[near_vertex] + adjmatrix[near_vertex][v]
                    parent[v] = near_vertex
    return dist, parent


def adj_matrix(path):

    df = pd.DataFrame()
    files = [path]

    for f in files:
        df = df.append(other=pd.read_csv(f), ignore_index=True)
    df.drop(['Coordinates', 'Intersection_Name'], axis=1, inplace=True)
    vertices = df['NodeID'].unique()

    adj = [[float("inf") for i in range(len(vertices))] for j in range(len(vertices))]
    
    for v in vertices:
        adj[v][v] = 0

    for index, row in df.iterrows():
        adj[row['NodeID']][row['ConnectedNodeID']] = row['Distance']

    return adj, vertices


def get_memory(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_memory(v, seen) for v in obj.values()])
        size += sum([get_memory(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_memory(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_memory(i, seen) for i in obj])
    return size


if __name__ == "__main__":

    x = []
    y = []
    x_size = []
    y_size = []
    for i in range(1, 15):
        print('Processing file', i)
        path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(i) + ".csv"
        start = time.time()
        matrix, vertices = adj_matrix(path)
        
        for v in tqdm(vertices):
            dist_list, parent_dict = dijkstra(v,matrix,vertices)

        end = time.time()
        x.append(i)
        y.append(end - start)
        x_size.append(i)
        y_size.append(get_memory(matrix))

    plt.plot(x, y)
    plt.title("Djikstra Matrix Time Comparison")
    plt.xticks([i for i in range(1, 15)])
    plt.xlabel("File number")
    plt.ylabel("Time (sec)")
    plt.savefig("Project2_Djikstra_2DArray_Time.png")

    plt.clf()
    plt.close()
    plt.plot(x_size, y_size)
    plt.title("Djikstra Matrix Memory Utilization")
    plt.xticks([i for i in range(1, 15)])
    plt.ylabel("Memory (bytes)")
    plt.xlabel("File number")
    plt.savefig("Project2_Djikstra_2DArray_Memory.png")

    print('\nPerformance (Seconds) ')
    for i, val in enumerate(y):
        print(f'File {i+1}: {val}')

    print('\nMemory utilization (Bytes)')
    for i, val in enumerate(y_size):
        print(f'File {i+1}: {val}')


# In[ ]:


#test cases
path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(4) + ".csv"
matrix, vertices = adj_matrix(path)

dist1, parent1 = dijkstra(192,matrix,vertices)
result1 = get_path(192, 163, parent1)
print('\nTEST CASE I')
print(f"Shortest distance between nodes 192 and 163: {dist1[163]}")
print(f"Shortest path from nodes 192 to 163: \n" + " -> ".join(map(str, result1)))

dist2, parent2 = dijkstra(138,matrix,vertices)
result2 = get_path(138, 66, parent2)
print('\nTEST CASE II')
print(f"Shortest distance between nodes 138 and 66: {dist2[66]}")
print(f"Shortest path from nodes 138 to 66: \n" + " -> ".join(map(str, result2)))

dist3, parent3 = dijkstra(465,matrix,vertices)
result3 = get_path(465, 22, parent3)
print('\nTEST CASE III')
print(f"Shortest distance between nodes 465 and 22: {dist3[22]}")
print(f"Shortest path from nodes 465 to 22: \n" + " -> ".join(map(str, result3)))


# In[ ]:




