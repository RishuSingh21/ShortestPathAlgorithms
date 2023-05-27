#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import gc


# In[28]:



class Node:
    def __init__(self, node, weight=None, next=None):
        self.node = node
        self.weight = weight
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, node, weight):
        new_vertex = Node(node, weight)
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
    vertices = sorted(df['NodeID'].unique())

    adj_list = []
    for v in vertices:
        newlist = LinkedList()
        for index, row in df.iterrows():
            if v == row['NodeID']:
                newlist.add_node(row['ConnectedNodeID'], row['Distance'])
        adj_list.append(newlist)
    
    adj = [[float("inf") for i in range(len(adj_list))] for j in range(len(adj_list))]

    for i in range(len(adj_list)):

        ll = adj_list[i]
        current = ll.head
        adj[i][i] = 0

        while current != None:
            node = current.node
            distance = current.weight
            adj[i][node] = distance
            current = current.next


    return adj, vertices




def get_path(source, destination, sd_path):
    result = []
    if sd_path[source][destination] == destination:
        return [source, destination]
    else:
        result += get_path(source, sd_path[source][destination],
                             sd_path)
        result += get_path(result[-1], destination, sd_path)[1:]

    return result


def floyd(matrix, vertices):
    sd_path = {}

    for k in tqdm(vertices):
        for i in vertices:
            for j in vertices:

                if i not in sd_path:
                    sd_path[i] = {}

                if matrix[i][j] > matrix[i][k] + matrix[k][j]:
                    matrix[i][j] = matrix[i][k] + matrix[k][j]
                    sd_path[i][j] = k
                else:
                    if i != j and matrix[i][j] != float("inf") and j not in sd_path[i]:
                        sd_path[i][j] = j

    return matrix, sd_path


def get_memory_bytes(obj, seen=None):
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
        size += sum([get_memory_bytes(v, seen) for v in obj.values()])
        size += sum([get_memory_bytes(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_memory_bytes(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_memory_bytes(i, seen) for i in obj])
    return size




# In[29]:


if __name__ == "__main__":

    x = []
    y = []
    x_size = []
    y_size = []
    for i in range(1, 15):
        path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(i) + ".csv"
        start = time.time()
        adj, vertices = adj_matrix(path)
        sd_matrix, sd_path = floyd(adj, vertices)
        end = time.time()
        x.append(i)
        y.append(end - start)
        x_size.append(i)
        y_size.append(get_memory_bytes(adj))


    plt.plot(x, y)
    plt.title("Floyd Linked List Time Comparison")
    plt.xticks([i for i in range(1, 15)])
    plt.xlabel("File number")
    plt.ylabel("Time (sec)")
    plt.savefig("Project2_Floyd_LinkedList_Time.png")

    plt.clf()
    plt.close()
    plt.title("Floyd Linked List Memory Utilization")
    plt.plot(x_size, y_size)
    plt.xticks([i for i in range(1, 15)])
    plt.xlabel("File number")
    plt.ylabel("Memory (bytes)")
    plt.savefig("Project2_Floyd_LinkedList_Memory.png")

    print('\nPerformance (Seconds) ')
    for i, val in enumerate(y):
        print(f'File {i+1}: {val}')

    print('\nMemory utilization (Bytes)')
    for i, val in enumerate(y_size):
        print(f'File {i+1}: {val}')


# In[30]:


path = "/Users/rishusingh/Dropbox/Mac/Documents/Upitt/5th Sem/Project2/Project2_Input_Files/Project2_Input_File" + str(4) + ".csv"
matrix, vertices = adj_matrix(path)
sd_matrix, sd_path = floyd(matrix, vertices)

result1 = get_path(192, 163, sd_path)
print('\nTEST CASE I')
print(f"Shortest distance between nodes 192 and 163 is: {sd_matrix[192][163]}")
print(f"Shortest path traversed from nodes 192 to 163 is: \n" + " -> ".join(map(str, result1)))


result2 = get_path(138, 66, sd_path)
print('\nTEST CASE II')
print(f"Shortest distance between nodes 138 and 66 is: {sd_matrix[138][66]}")
print(f"Shortest path traversed from nodes 138 to 66 is: \n" + " -> ".join(map(str, result2)))


result3 = get_path(465, 22, sd_path)
print('\nTEST CASE III')
print(f"Shortest distance between nodes 465 and 22 is: {sd_matrix[465][22]}")
print(f"Shortest path traversed from nodes 465 to 22 is: \n" + " -> ".join(map(str, result3)))


# In[ ]:





# In[ ]:




