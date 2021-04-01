#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import math
import time
import copy
import pandas as pd
import operator
from collections import defaultdict
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt


# In[ ]:


class Node(object):
    def __init__(self, data = None, next_node = None, prev_node = None):
        self.data = data
        self.next_node =  next_node
        self.prev_node = prev_node

class DoublyLinkedList(object):
    def __init__(self, head = None):
        self.head = head
    
    def traverse(self):
        current_node = self.head
        while current_node != None:
            print(current_node.data)
            current_node = current_node.next_node
    
    def get_size(self):
        count = 0
        current_node = self.head
        while current_node != None:
            count += 1
            current_node = current_node.next_node
        return count
            
    def append(self, data):
        new_node = Node(data)
        current_node = self.head
        new_node.next_node = current_node
        new_node.prev_node = None
        if current_node != None:
            current_node.prev_node = new_node
        self.head = new_node
    
    def insert_end(self, data):
        new_node = Node(data)
        new_node.next = None
        if self.head == None:
            new_node.prev_node = None
            self.head = new_node
        return
    
        first_node = self.head
        while first_node.next_node:
            first_node = first_node.next_node
        first_node.next_node = new_node
        new_node.prev_node = first_node
    
    
    def delete(self, data):
        current_node = self.head
        while current_node != None:
            if current_node.data == data and current_node == self.head:
                if not current_node.next_node:
                    current_node = None
                    self.head = None
                    return
                else:
                    q = current_node.next_node
                    current_node.next_node = None
                    q.prev_node = None
                    current_node = None
                    self.head = q
                    return
            
            elif current_node.data == data:
                #print("hello")
                if current_node.next_node != None:
                    p = current_node.prev_node
                    q = current_node.next_node
                    p.next_node = q
                    q.prev_node = p
                    current_node.next_node = None
                    current_node.prev_node = None
                    current_node = None
                    return
                else:
                    #print("bye")
                    p = current_node.prev_node
                    p.next_node = None
                    current_node.prev_node = None
                    current = None
                    return
            current_node =  current_node.next_node
    
class Bucket(object):
    def __init__(self, value):
        self.data = DoublyLinkedList()

class Bucket_Arrays(object):
    def __init__(self, maxdegree):
        self.bucket_range = maxdegree
        self.left_buckets = np.array([Bucket(i) for i in range(-self.bucket_range,self.bucket_range+1)])
        self.right_buckets = np.array([Bucket(i) for i in range(-self.bucket_range,self.bucket_range+1)])
        self.max_gain_left = -maxdegree
        self.max_gain_right = -maxdegree
        self.left_size = 0
        self.right_size = 0
        self.fixed = 0


# In[ ]:


def calculate_num_cuts(df):
    connected_edges = df[df['Partition']==0]['ID connected vertices'].values
    outside_partition = list(df[df['Partition']==1].index)
    connected_edges = [item in outside_partition for sublist in connected_edges for item in sublist]
    num_cuts = np.sum(connected_edges)
    return num_cuts

def initialize_gain_buckets(df, buckets):
    max_degree = 16
    for i in range(len(df)):
        connected_edges = df['ID connected vertices'][i]
        partition_value = df['Partition'][i]
        gain = 0
        for j in connected_edges:
            connected_partition = df['Partition'][j]
            if(partition_value != connected_partition):
                gain += 1
            else:
                gain -= 1
        df.loc[i,'Gain'] = gain
        
        if partition_value == 0:
            buckets.left_buckets[gain+max_degree].data.append(i)
            buckets.left_size += 1
            if(buckets.max_gain_left < gain):
                buckets.max_gain_left = gain
        else:
            buckets.right_buckets[gain+max_degree].data.append(i)
            buckets.right_size += 1
            if(buckets.max_gain_right < gain):
                buckets.max_gain_right = gain
    return df,buckets

def update_df_buckets(df,buckets,vertex_max_gain, gain_update,max_degree):

    partition_value = df.loc[vertex_max_gain,'Partition']
    df.loc[vertex_max_gain,'Partition'] = int(not(partition_value))
    if df.loc[vertex_max_gain,'Partition'] == 1:
        buckets.left_buckets[gain_update+max_degree].data.delete(vertex_max_gain)
        buckets.left_size -= 1
        while(buckets.left_buckets[buckets.max_gain_left+max_degree].data.head == None and buckets.max_gain_left>-max_degree):
            buckets.max_gain_left -= 1
        df,buckets = re_calculate_gain(df, buckets,vertex_max_gain,max_degree,1)
        
    else:
        buckets.right_buckets[gain_update+max_degree].data.delete(vertex_max_gain)
        buckets.right_size -= 1
        while(buckets.right_buckets[buckets.max_gain_right+max_degree].data.head == None and buckets.max_gain_right>-max_degree):
            buckets.max_gain_right -= 1
        df,buckets = re_calculate_gain(df, buckets,vertex_max_gain,max_degree,0)
    return df, buckets

def re_calculate_gain(df, buckets, vertex_max_gain, max_degree, changed_partition):
    vertices_to_update = df.loc[vertex_max_gain]['ID connected vertices']
    for i in vertices_to_update:
        if(df.loc[i]['Fixed'] == 0):
            current_gain = df.loc[i]['Gain']
            current_partition = df.iloc[i]['Partition']
            if current_partition == 1:
                buckets.right_buckets[current_gain+max_degree].data.delete(i)
                buckets.right_size -= 1
                while(buckets.right_buckets[buckets.max_gain_right+max_degree].data.head == None and buckets.max_gain_right>-max_degree):
                    buckets.max_gain_right -= 1
            else:
                buckets.left_buckets[current_gain+max_degree].data.delete(i)
                buckets.left_size -= 1
                while(buckets.left_buckets[buckets.max_gain_left+max_degree].data.head == None and buckets.max_gain_left>-max_degree):
                    buckets.max_gain_left -= 1
        
            partition_value = df['Partition'][i]
            if (partition_value == changed_partition):
                new_gain = current_gain - 2
            else:
                new_gain = current_gain + 2
            df.loc[i,'Gain'] = new_gain
            if(partition_value == 1): 
                buckets.right_buckets[new_gain+max_degree].data.append(i)
                buckets.right_size += 1
                if(buckets.max_gain_right < new_gain):
                    buckets.max_gain_right = new_gain
            else:
                buckets.left_buckets[new_gain+max_degree].data.append(i)
                buckets.left_size += 1
                if(buckets.max_gain_left < new_gain):
                    buckets.max_gain_left = new_gain
    df.loc[vertex_max_gain,'Fixed'] = 1
    buckets.fixed += 1
    return df, buckets

def initialise_data(partitioning):
    # Loading the single planar graph of 500 vertices
    data = defaultdict(list)
    for line in open("Graph500.txt"):
        split_line=line.split()
        ID_vertex = int(split_line[0])
        num_connected_vertices  = int(split_line[2])
        ID_connected_vertices = [int(i)-1 for i in split_line[3:]]
        if (ID_vertex) not in data.keys():
            data[ID_vertex].append(0)
            data[ID_vertex].append(0)
            data[ID_vertex].append(ID_connected_vertices)
            data[ID_vertex].append(0)
    data_frame = pd.DataFrame(data.values(),columns = ['Gain', 'Fixed','ID connected vertices', 'Partition'])

    num_vertices = len(data_frame)
    if(partitioning is None):
        partition = random.sample(range(0,num_vertices),250)
        data_frame.loc[partition,'Partition'] = 1
    else:
        data_frame["Partition"] = partitioning
    return data_frame

def FM_one_pass(df, current_num_cuts):
    num_cuts = current_num_cuts
    num_vertices = 500
    min_cuts = num_cuts
    max_degree = 16
    buckets = Bucket_Arrays(max_degree)
    df,buckets = initialize_gain_buckets(df, buckets)
    save_partition = copy.deepcopy(df['Partition'].values)
    while(buckets.fixed < num_vertices):
        if(buckets.left_size >= buckets.right_size):
            gain_update = buckets.max_gain_left
            vertex_max_gain = buckets.left_buckets[gain_update+max_degree].data.head.data
            df, buckets = update_df_buckets(df,buckets,vertex_max_gain,gain_update,max_degree)
            
        else:
            gain_update = buckets.max_gain_right
            vertex_max_gain = buckets.right_buckets[gain_update+max_degree].data.head.data
            df, buckets = update_df_buckets(df,buckets,vertex_max_gain,gain_update,max_degree)
            
        num_cuts = num_cuts - gain_update
        if(num_cuts < min_cuts and buckets.left_size == buckets.right_size):
            save_partition = copy.deepcopy(df['Partition'].values)
            min_cuts = num_cuts
    return min_cuts, save_partition

def mutation(bitstring, r_mut):
    num_ones_flipped = 0
    num_zeros_flipped = 0
    for i in range(len(bitstring)):
        if random.random() < r_mut:
            if(bitstring[i]) == 1:
                num_ones_flipped +=1
            else:
                num_zeros_flipped +=1
            bitstring[i] = int(not(bitstring[i]))
    
    while(num_ones_flipped != num_zeros_flipped):
        if (num_ones_flipped > num_zeros_flipped):
            to_flip = random.randint(0, len(bitstring)-1)
            if (bitstring[to_flip] == 0):
                bitstring[to_flip] = int(not(bitstring[to_flip]))
                num_zeros_flipped += 1
        else:
            to_flip = random.randint(0, len(bitstring)-1)
            if (bitstring[to_flip] == 1):
                bitstring[to_flip] = int(not(bitstring[to_flip]))
                num_ones_flipped += 1
    return bitstring

def initialize_partition(num_vertices):
    partition = np.ones(num_vertices)
    partition[:250] = 0
    np.random.shuffle(partition)
    return partition

def initialize_population(population_size,num_vertices):
    population_array = np.array([]).reshape(0,num_vertices)
    for i in range(population_size):
        partition = initialize_partition(num_vertices)
        population_array = np.vstack((population_array,partition))
    return population_array

def uniform_crossover(parent_1, parent_2):
    chromosome_length = len(parent_1)
    child = np.zeros(chromosome_length)
    remaining_pos = np.array([])
    for i in range(chromosome_length):
        # Children inherit bits that parents agree on 
        if parent_1[i] == parent_2[i]:     
            child[i] = parent_1[i]
        else:
            remaining_pos = np.append(remaining_pos,i)
            
    remaining_pos = list(remaining_pos)
    num_ones = int(np.sum(child))
    required_ones = int(chromosome_length//2) - num_ones
    random_sample = random.sample(remaining_pos,required_ones)
    for j in random_sample:
        child[int(j)] = 1
        
    return child

def FM_one_run(max_passes,optimal_partition):
    flag = 0
    total_passes = 0
    total_time = 0
    best_local_optimum = math.inf
    while(True):
        df = initialise_data(optimal_partition)
        current_num_cuts = calculate_num_cuts(df)
        start = time.time()
        local_optimum, optimal_partition = FM_one_pass(df,current_num_cuts)
        total_passes += 1
        end = time.time()
        elpased_time = end - start
        total_time += elpased_time

        if(local_optimum < best_local_optimum and total_passes <= max_passes):
            best_local_optimum = local_optimum
            best_partition = optimal_partition
            flag = 0
        else:
            flag = 1
        if(flag == 1):
            break
    converged_local_optimum = best_local_optimum
    converged_partition = best_partition
    return [converged_local_optimum, converged_partition, total_passes] 


# In[ ]:


def MLS_one_run(max_passes):
    flag = 0
    total_passes = 0
    total_time = 0
    best_local_optimum = math.inf
    mls_local_optimum = math.inf
    optimal_partition = None
    run_data_frame = pd.DataFrame(columns = ['Coverged local optima','Time(s)'])
    total_time = 0
    start = time.time()
    new_start = time.time()
    while(True):
        df = initialise_data(optimal_partition)
        current_num_cuts = calculate_num_cuts(df)
        
        local_optimum, optimal_partition = FM_one_pass(df,current_num_cuts)
        total_passes += 1
        
        if(local_optimum < best_local_optimum and total_passes <= max_passes):
            best_local_optimum = local_optimum
            flag = 0
        else:
            
            if(total_passes <= max_passes):
                optimal_partition = None
                if(mls_local_optimum > best_local_optimum): 
                    new_end = time.time()
                    total_time = new_end - new_start
                    mls_local_optimum = best_local_optimum
                best_local_optimum = math.inf
                flag = 0
            else:
                flag = 1
            end = time.time()
            elpased_time = end - start
            start = time.time()
            observations= [mls_local_optimum,total_time]
            run_data_frame.loc[len(run_data_frame)] = observations
        if(flag == 1):
            break
    return run_data_frame

def ILS_one_run(max_passes):
    flag = 0
    ils_run = 0
    total_passes = 0
    total_time = 0
    best_local_optimum = math.inf
    best_local_partition = np.array([])
    
    p_mut = 0.01
    ils_local_optimum = math.inf
    ils_partition = np.array([])
    
    ils_best_local_optimum = math.inf
    ils_old_partition = np.array([])
    ils_old_local_optimum = np.array([])
    optimal_partition = None
    run_data_frame = pd.DataFrame(columns = ['Coverged local optima','Num_roa','Time(s)'])
    start = time.time()
    num_roa = 0
    new_start = time.time()
    elapsed_time = 0
    while(True):
        df = initialise_data(optimal_partition)
        current_num_cuts = calculate_num_cuts(df)
        local_optimum, optimal_partition = FM_one_pass(df,current_num_cuts)
        total_passes += 1
        
        if(local_optimum < best_local_optimum and total_passes <= max_passes):
            best_local_optimum = local_optimum
            best_local_partition = optimal_partition
            flag = 0
        else:
            if(total_passes <= max_passes):
                optimal_partition = None
                if(best_local_optimum < ils_local_optimum): 
                    new_end = time.time()
                    total_time = new_end - new_start
                    ils_local_optimum = best_local_optimum
                    ils_partition = best_local_partition
 
                    new_population = mutation(ils_partition,p_mut)
                    optimal_partition = copy.deepcopy(new_population)
                
                else:
                    if(best_local_optimum == ils_local_optimum):
                        num_roa += 1
                    new_population = mutation(ils_partition,p_mut)
                    optimal_partition = copy.deepcopy(new_population)
                
                best_local_optimum = math.inf
                flag = 0
            else:
                flag = 1
            end = time.time()
            elpased_time = end - start
            start = time.time()
            observations= [ils_local_optimum,num_roa,total_time]
            num_roa = 0
            run_data_frame.loc[len(run_data_frame)] = observations
        if(flag == 1):
            break
    return run_data_frame

def GLS_one_run_no_duplicate(max_passes):
    population_size = 50
    num_vertices = 500
    total_passes = 0
    total_time = 0
    population = initialize_population(population_size,num_vertices)
    population_local_optimum = np.zeros(population_size)
    start = time.time()
    new_start = time.time()
    for i in range(len(population)):
        optimal_partition = population[i]

        local_optimum,local_optimum_partition,num_passes = FM_one_run(max_passes,optimal_partition)
        total_passes += num_passes
        population_local_optimum[i] = local_optimum
        population[i] = local_optimum_partition
    while(total_passes < max_passes):

        choice_1, choice_2 = random.sample(range(0, population_size), 2)
        parent_1 = population[choice_1]
        parent_2 = population[choice_2]
        hamming_distance = np.count_nonzero(parent_1!=parent_2)

        if (hamming_distance > num_vertices//2):
            parent_to_invert = np.random.choice([1,2])
            if(parent_to_invert == 1):
                parent_1 = 1 - parent_1
            else:
                parent_2 = 1 - parent_2

        child = uniform_crossover(parent_1, parent_2)
        optimal_partition = child
        child_local_optimum,child_optimum_partition,num_passes = FM_one_run(max_passes,optimal_partition)
        total_passes += num_passes
        
        worst_solution_in_population = np.max(population_local_optimum)
        if(child_local_optimum <= worst_solution_in_population):
            flag = 0
            if(child_local_optimum in population_local_optimum):
                if(child_local_optimum == worst_solution_in_population):
                    flag = 0
                else:
                    flag = 1
                    
            if(flag == 0):    
                worst_solution_index = np.argmax(population_local_optimum)
                population[worst_solution_index] = child_optimum_partition
                population_local_optimum[worst_solution_index] = child_local_optimum

                end = time.time()
                total_time = end - start

        
        current_local_optima = np.min(population_local_optimum)
        observations = [current_local_optima,total_time]
           
    converged_local_optima = np.min(population_local_optimum)
    
    return [converged_local_optima,total_time,final_elpased_time]


# In[ ]:


def GLS_one_run(max_passes):
    population_size = 50
    num_vertices = 500
    total_passes = 0
    total_time = 0
    new_start = time.time()
    population = initialize_population(population_size,num_vertices)
    population_local_optimum = np.zeros(population_size)
    start = time.time()
    new_start = time.time()
    for i in range(len(population)):
        optimal_partition = population[i]

        local_optimum,local_optimum_partition,num_passes = FM_one_run(max_passes,optimal_partition)
        total_passes += num_passes
        population_local_optimum[i] = local_optimum
        population[i] = local_optimum_partition
    while(total_passes < max_passes):

        choice_1, choice_2 = random.sample(range(0, population_size), 2)
        parent_1 = population[choice_1]
        parent_2 = population[choice_2]
        hamming_distance = np.count_nonzero(parent_1!=parent_2)

        if (hamming_distance > num_vertices//2):
            parent_to_invert = np.random.choice([1,2])
            if(parent_to_invert == 1):
                parent_1 = 1 - parent_1
            else:
                parent_2 = 1 - parent_2

        child = uniform_crossover(parent_1, parent_2)
        optimal_partition = child
        child_local_optimum,child_optimum_partition,num_passes = FM_one_run(max_passes,optimal_partition)
        total_passes += num_passes
        
        worst_solution_in_population = np.max(population_local_optimum)
        if(child_local_optimum <= worst_solution_in_population):
            flag = 0
        else:
            flag = 1
                    
        if(flag == 0):    
            worst_solution_index = np.argmax(population_local_optimum)
            population[worst_solution_index] = child_optimum_partition
            population_local_optimum[worst_solution_index] = child_local_optimum

            end = time.time()
            total_time = end - start

        
        current_local_optima = np.min(population_local_optimum)
        observations = [current_local_optima,total_time]

    converged_local_optima = np.min(population_local_optimum)
    
    return [converged_local_optima,total_time,final_elpased_time]


# In[ ]:


# same run time
max_passes = 10000
total_runs = 25
mls_run_data_frame = pd.DataFrame(columns = ['Converged local optima','Num_local_optima','Time(s)'])
ils_run_data_frame = pd.DataFrame(columns = ['Converged local optima','Num_roa','Num_local_optima','Time(s)'])
gls_run_data_frame = pd.DataFrame(columns = ['Converged local optima','Time(s)'])
gls_no_duplicate_run_data_frame = pd.DataFrame(columns = ['Converged local optima','Time(s)'])
for i in range(0,total_runs):
    observations, mls_time = MLS_one_run(max_passes)
    mls_observations = [observations.min()[0],len(observations),observations.max()[1],mls_time]
    print("MLS",mls_observations)
    mls_run_data_frame.loc[len(mls_run_data_frame)] = mls_observations

for i in range(0,total_runs):
    observations, ils_time = ILS_one_run(max_passes)
    ils_observations = [observations.min()[0],observations.sum()[1],len(observations),observations.max()[2]]
    print("ILS",ils_observations)
    ils_run_data_frame.loc[len(ils_run_data_frame)] = ils_observations
    

for i in range(0,total_runs):
    observations = GLS_one_run(max_passes)
    best_local_optimum = observations[0]
    total_time = observations[1]
    gls_observations = [best_local_optimum,total_time]
    print("GLS_no_duplicate",gls_observations)
    gls_run_data_frame.loc[len(gls_run_data_frame)] = gls_observations

for i in range(0,total_runs):
    observations = GLS_one_run_no_duplicate(max_passes)
    best_local_optimum = observations[0]
    total_time = observations[1]
    gls_observations = [best_local_optimum,total_time]
    print("GLS_normal",gls_observations)
    gls_no_duplicate_run_data_frame.loc[len(gls_run_data_frame)] = gls_observations

