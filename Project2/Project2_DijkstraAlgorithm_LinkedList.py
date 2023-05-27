#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys


# In[2]:


def get_path(source, end, parent):
    short_path = [end]

    while source != end:
        short_path.append(parent[end])
        end = parent[end]

    short_path.reverse()
    return short_path


class Graph:
    def __init__(self, node, weight=None, next=None):
        self.node = node
        self.weight = weight
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, node, weight):
        new_vertex = Graph(node, weight)
        if self.head:
            current_vertex = self.head
            while current_vertex.next:
                current_vertex = current_vertex.next
            current_vertex.next = new_vertex
        else:
            self.head = new_vertex


def adj_matrix(path):
    df = pd.DataFrame()
    files = [path]
    for f in files:
        df = df.append(other=pd.read_csv(f), ignore_index=True)
    df.drop(['Coordinates', 'Intersection_Name'], axis=1, inplace=True)
    vertices =df['NodeID'].unique()
    adj_list = []
    for v in vertices:
        mylist = LinkedList()
        for index, row in df.iterrows():
            if v == row['NodeID']:
                mylist.add_node(row['ConnectedNodeID'], row['Distance'])
        adj_list.append(mylist)

    return adj_list


def min_dist(adj_list, is_visited, dist):
    k = 0
    min_d = float("inf")

    for v,_ in enumerate(adj_list):
        if dist[v] < min_d and is_visited[v] == False:
            min_d = dist[v]
            k = v
    return k


def dijkstra(source, adj_list):
    dist = {}
    is_visited = {}

    for v,i in enumerate(adj_list):
        dist[v] = float("inf")
        is_visited[v] = False

    dist[source] = 0
    parent = {source: -1}

    for index, i in enumerate(adj_list):

        near_vertex = min_dist(adj_list, is_visited, dist)
        is_visited[near_vertex] = True

        ll = adj_list[near_vertex]

        current_vertex = ll.head
        while current_vertex != None:
            neighbor = current_vertex.node
            wt = current_vertex.weight
            current_vertex=current_vertex.next
            if not is_visited[neighbor]:
                if dist[neighbor] > dist[near_vertex] + wt:
                    dist[neighbor] = dist[near_vertex] + wt
                    parent[neighbor] = near_vertex

    return dist,parent


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



# In[3]:


if __name__ == "__main__":

    x = []
    y = []
    x_size = []
    y_size = []
    for i in range(1, 15):
        print('Processing file', i)
        path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(i) + ".csv"
        start = time.time()
        adj= adj_matrix(path)
        for v in tqdm(range(len(adj))):
            dist, parent = dijkstra(v, adj)
        end = time.time()
        x.append(i)
        y.append(end - start)
        x_size.append(i)
        y_size.append(get_memory(adj))


    plt.plot(x, y)
    plt.xlabel("File number")
    plt.xticks([i for i in range(1, 15)])
    plt.ylabel("Time (sec)")
    plt.title("Djikstra Linked List Time Comparison")
    plt.savefig("Djikstra_Linked_List_Time.png")

    plt.clf()
    plt.close()
    plt.plot(x_size, y_size)
    plt.xlabel("File number")
    plt.xticks([i for i in range(1, 15)])
    plt.ylabel("Memory (bytes)")
    plt.title("Djikstra Linked List Memory Utilization")
    plt.savefig("Djikstra_Linked_List_Memory.png")

    print('\nPerformance (Seconds) ')
    for i, val in enumerate(y):
        print(f'File {i+1}: {val}')

    print('\nMemory utilization (Bytes)')
    for i, val in enumerate(y_size):
        print(f'File {i+1}: {val}')


    


# In[4]:


path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(4) + ".csv"
adj= adj_matrix(path)

dist1,path1 = dijkstra(192,adj)
result1 = get_path(192, 163, path1)
print('\nTEST CASE I')
print(f"Shortest distance between nodes 192 and 163 is: {dist1[163]}")
print(f"Shortest path traversed from nodes 192 to 163 is: \n" + " -> ".join(map(str, result1)))

dist2,path2 = dijkstra(138, adj)
result2 = get_path(138,66,path2)
print('\nTEST CASE II')
print(f"Shortest distance between nodes 138 and 66 is: {dist2[66]}")
print(f"Shortest path traversed from nodes 138 to 66 is: \n" + " -> ".join(map(str, result2)))

dist3,path3 = dijkstra(465, adj)
result3 = get_path(465, 22, path3)
print('\nTEST CASE III')
print(f"Shortest distance between nodes 465 and 22 is: {dist3[22]}")
print(f"Shortest path traversed from nodes 465 to 22 is: \n" + " -> ".join(map(str, result3)))


# In[ ]:




