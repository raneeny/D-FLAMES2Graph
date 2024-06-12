
import numpy as np
from tensorflow import keras
import pandas
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
from scipy.spatial import distance
import matplotlib.pyplot as plt
import tensorflow as tf
from random import seed, shuffle
import matplotlib.pyplot as plt
import pickle
import math
import pandas as pd
import networkx as nx


class ServerGraphAggregation:
    def __init__(self):  
        self = self
    
    """ Function to calculate Euclidean distance between two arrays """
    def euclidean_distance(self, arr1, arr2):
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        return np.linalg.norm(arr1 - arr2)
    
    """ Function to perform min-max normalization of a vector"""
    def normalize_vector(self, vector, min_value, max_value):
        range_value = max_value - min_value
        normalized_vector = [(x - min_value) / range_value for x in vector]
        return normalized_vector
    

    """ Function to update the centroids of the new nodes. New centroids are equal to the average of the
    centroids merged into the same node.  It return a dict containing the new mapping (name : centroid_value)"""
    def update_centroids(self, G):

        # obtaining the set of node names of all the clients
        node_names = set()
        for i in range(len(G)):
            node_names.update(G[i].nodes)
        node_names = list(node_names)

        new_centroids = []
        for n in node_names: # for each node name, calculate the average of the centroid values of all the nodes with this name
            n_centroids = []
            for i in range(len(G)):
                node_temp = list(G[i].nodes)
                if n in node_temp:
                    centroid = G[i].nodes[n]['CentroidValues']
                    n_centroids.append(centroid)
            medium = np.mean(n_centroids,axis=0)
            new_centroids.append(medium)

        new_centroids_dict = dict(zip(node_names,new_centroids))
        return new_centroids_dict
    

    """ Function to aggregate graphs of different clients, on the server side. 
    Returns the aggregated server graph, with and without attributes, 
    as well as the list of centroids"""
    def aggregate_graphs_clients(self, G, centroid_list, threshold=0.75):
        print('Staring server graph aggregation')

        # Adjusting Graphs
        for i in range(len(G)): 
                
            nodes = list(G[i].nodes)
            for node in nodes:
                #print(f'Node name {node}')
                layer = int(node.split()[0][-1])  # Extract the layer number from the node name
                G[i].nodes[node]['Layer'] = layer

                centroid = int(node.split()[1])  # Extract centroid ID
                G[i].nodes[node]['Centroid'] = centroid 

                centroid_values = centroid_list[i][layer][centroid]  # Extract centroid values
                G[i].nodes[node]['CentroidValues'] = centroid_values 
    



        layer_view = dict(G[0].nodes.data('Layer'))
        layers = set(layer_view.values())

        # working on layer 0, then defining a loop (TODO)

        n_clients = len(G)
        #print(f'number of clients: {n_clients}')

        for l in layers:
            #print('Working on layer ' + str(l))
            # finding number of nodes on the layer, for each client
            num_nodes = []
            for i in range(n_clients):
                nodes_layer = [node for node, data in G[i].nodes(data=True) if data.get('Layer') == l]
                num_nodes.append(len(nodes_layer))

            # defining max number of potential nodes
            min_generated = min(num_nodes)
            max_generated = int(np.array(num_nodes).mean())
            #max_generated = min(num_nodes) + 1
            #max_generated = max(num_nodes)  
            #max_generated = int(np.median(np.array(num_nodes)))
            my_max = max_generated
            #my_max = math.floor((max_generated + min_generated)/2) # we set an average value so as to avoid the number of nodes exploding
            #print('max number of nodes to generate for layer ' + str(l) + ' : ' + str(my_max))

            # we use the client with the minimum number of nodes on the layer as point of reference
            min_index = num_nodes.index(min(num_nodes)) # TODO consider the case where more clients have the maximum value
            ref = G[min_index]
            ref_nodes = [(node, data['CentroidValues']) for node, data in ref.nodes(data=True) if data.get('Layer') == l]
            ref_centroids = list(dict(ref_nodes).values())

            cents_to_merge = []
            test_merged = []
            test_centroids = []
            clients_id = []
            nodes_id = []

            # create a dataframe with all the centroids and their informations
            for i in range(n_clients): 
                if (i!=min_index): # iterating an all the OTHER clients
                        test_graph = G[i]
                        test_nodes_i = [(node, data['CentroidValues']) for node, data in test_graph.nodes(data=True) if data.get('Layer') == l]
                        nodes_id.append(list(dict(test_nodes_i).keys()))
                        test_centroids_i = list(dict(test_nodes_i).values())
                        test_centroids.append(test_centroids_i)
                        test_merged_i = [False] * len(test_centroids_i)
                        clients_id_i = [i] * len(test_centroids_i)
                        clients_id.append(clients_id_i)

            flattened_centroids = [value for sublist in test_centroids for value in sublist]
            flattened_clients_id = [value for sublist in clients_id for value in sublist]
            flattened_nodes_id = [value for sublist in nodes_id for value in sublist]

            df_others = pd.DataFrame(list(zip(flattened_centroids,flattened_clients_id,flattened_nodes_id)),columns=['Centroid','ClientID','NodesID'])
            df_others['Distances'] = ''
            df_others['Merged'] = ''
            df_others['MergeWith'] = ''
            new_names = []
            distances = []
            merged = []
            mergwith = []
            max_dist = []

            for index, row in df_others.iterrows(): # for each node of the OTHER clients
                # calculate euclidean distances between the node and all the reference ones
                distances_i = [self.euclidean_distance(row['Centroid'], arr) for arr in ref_centroids]
                #print(f'distances_i {distances_i}' )
                distances.append(distances_i)
                max_dist_i = max(distances_i)
                max_dist.append(max_dist_i)

            # Normalize distances
            # Flatten the vector of distances
            #print(f'distances {distances}')
            flattened_vector = np.concatenate(distances)
            # Find the maximum and minimum values
            max_value = np.max(flattened_vector)
            min_value = np.min(flattened_vector)
            normalized_vector = np.array(self.normalize_vector(flattened_vector,min_value,max_value))
            distances = normalized_vector.reshape(-1, len(ref_centroids)).tolist()
            normalized_max = self.normalize_vector(max_dist,min_value,max_value)
            df_others['Distances'] = distances
            df_others['MaxDist'] = normalized_max

            for index, row in df_others.iterrows():
                # check if the node is mergeable, based on the threshold
                distances_i = row['Distances']
                merged_i = any(distance <= threshold for distance in distances_i)
                merged.append(merged_i)

                if(merged_i == True):
                    mergwith_i = np.argmin(distances_i)
                    new_name_i = f'layer{l} {mergwith_i}' # marked if mergeable
                else:
                    mergwith_i = ""
                    new_name_i = ""

                mergwith.append(mergwith_i)
                new_names.append(new_name_i)


            #df_others['Distances'] = distances
            df_others['Merged'] = merged   
            df_others['MergeWith'] = mergwith
            #df_others['MaxDist'] = max_dist
            df_others['NewName'] = new_names

            num_not_merged = (df_others['Merged'] == False).sum()
            num_nodes_add = my_max - min_generated # how many nodes I can still add

            
            if num_not_merged <= num_nodes_add: # if it is possible to add all the not-merged nodes as new reference ones
                df_others['NewNode'] = df_others['Merged'] == False # add all the not-merged nodes as new reference ones
                df_others['Merged'] = True # Solved for all the nodes

                # generating new names for the new reference ones
                unique_count = min_generated
                for index, row in df_others[df_others['NewNode'] == True].iterrows():
                    df_others.at[index, 'NewName'] = f'layer{l} {unique_count}'
                    unique_count += 1

            else: # only num_nodes_add can be actually added 
                # selecting the num_nodes_add least distant nodes as new nodes to add
                least_dist = df_others.nsmallest(num_nodes_add,'MaxDist')
                df_others['NewNode'] = False
                df_others.loc[least_dist.index,'NewNode'] = True 
                df_others.loc[least_dist.index,'Merged'] = True # Solved for these nodes

                # generating new names for the new reference ones
                for i, index in enumerate(least_dist.index):
                    df_others.at[index, 'NewName'] = f'layer{l} {i + min_generated}'
            
            # check if there are any other not-merged nodes
            if any(df_others['Merged'] == False):

                # extend the list of reference nodes (new reference nodes + old reference ones)
                new_nodes = df_others[df_others['NewNode'] == True] 
                new_centroids = new_nodes['Centroid'].tolist()
                new_ref_centroids = ref_centroids.copy()
                new_ref_centroids.extend(new_centroids) 

                # for each not-merged node
                for index, row in df_others.iterrows():
                    if row['Merged'] == False: # here normalization is not needed, the merging is not based on a threshold
                        # calculate the distance
                        new_dist = [self.euclidean_distance(row['Centroid'], arr) for arr in new_ref_centroids]
                        df_others.at[index,'Distances'] = new_dist
                        mergwith_i = np.argmin(new_dist)
                        df_others.at[index,'MergeWith'] = mergwith_i
                        df_others.at[index,'Merged'] = True
                        df_others.at[index,'NewName'] = f'layer{l} {mergwith_i}'

            # update node names for reference client
            old_names = [ref[0] for ref in ref_nodes]
            update_names = [f'layer{l} {i}' for i in range(len(old_names))]
            node_mapping = dict(zip(old_names, update_names))
            #nx.relabel_nodes(G[max_index], node_mapping, copy=False)
            G[min_index] = nx.relabel_nodes(G[min_index], node_mapping, copy=True)
            # update node names for all the other clients
            for i in range(n_clients): 
                if (i!=min_index): # iterating an all the OTHER clients
                        df_temp = df_others[df_others['ClientID']==i]
                        node_mapping = dict(zip(df_temp['NodesID'], df_temp['NewName']))
                        G[i] = nx.relabel_nodes(G[i], node_mapping, copy=True)

        G_new = G
        # end of the loop -> already worked on all the layers
        # calculate centroids value            
        new_centroids_dict = self.update_centroids(G)
        #print('centroid dict')
        #print(new_centroids_dict)
        # aggregate client graphs into server graph
        server_graph = nx.compose_all(G)
        # update centroid values of server graph
        nx.set_node_attributes(server_graph, new_centroids_dict, 'CentroidValues')

        layers = [0,1,2]
        n = 3
        final_cents = []
        for l in layers:
            nod_layer = [(node, data['CentroidValues']) for node, data in server_graph.nodes(data=True) if data.get('Layer') == l]
            my_dict = dict(nod_layer)
            m = len(my_dict)
            layer_cents = [[] for i in range(m)]
            for name, centroid in my_dict.items():
                index2 = int(name.split()[1])
                layer_cents[index2] = centroid
            final_cents.append(layer_cents)

        #print(f'final centroids {final_cents}')

        server_graph_without = server_graph.copy()
        for node in server_graph_without:
            server_graph_without.nodes[node].pop('Layer')
            server_graph_without.nodes[node].pop('Centroid')
            server_graph_without.nodes[node].pop('CentroidValues')
        

        return server_graph, server_graph_without, final_cents
    

    """ Function to aggregate the graph of a client with the server's one. 
    Returns the aggregated graph, with and without attributes, as well as the list of centroids """
    def aggregate_graphs_server(self, G, centroid_list, threshold=0.75):
        print('Staring server graph aggregation')

        # Adjusting Graphs
        for i in range(len(G)): 

            nodes = list(G[i].nodes)
            for node in nodes:
                #print(f'Node name {node}')
                layer = int(node.split()[0][-1])  # Extract the layer number from the node name
                G[i].nodes[node]['Layer'] = layer

                centroid = int(node.split()[1])  # Extract centroid ID
                G[i].nodes[node]['Centroid'] = centroid 

                centroid_values = centroid_list[i][layer][centroid]  # Extract centroid values
                G[i].nodes[node]['CentroidValues'] = centroid_values 




        layer_view = dict(G[0].nodes.data('Layer'))
        layers = set(layer_view.values())

        # working on layer 0, then defining a loop (TODO)

        n_clients = len(G)
        #print(f'number of clients: {n_clients}')

        for l in layers:
            #print('Working on layer ' + str(l))
            # finding number of nodes on the layer, for each client
            num_nodes = []
            for i in range(n_clients):
                nodes_layer = [node for node, data in G[i].nodes(data=True) if data.get('Layer') == l]
                num_nodes.append(len(nodes_layer))

            # defining max number of potential nodes
            max_generated = sum(num_nodes) # if there are no similarities, all the nodes are added as new nodes
            min_generated = max(num_nodes) # if all the nodes are similar, the number of nodes are the maximum among the clients
            my_max = math.floor((max_generated + min_generated)/2) # we set an average value so as to avoid the number of nodes exploding
            #print('max number of nodes to generate for layer ' + str(l) + ' : ' + str(my_max))

            # we use the client with the maximum number of nodes on the layer as point of reference
            max_index = num_nodes.index(max(num_nodes)) # TODO consider the case where more clients have the maximum value
            ref = G[max_index]
            ref_nodes = [(node, data['CentroidValues']) for node, data in ref.nodes(data=True) if data.get('Layer') == l]
            ref_centroids = list(dict(ref_nodes).values())

            cents_to_merge = []
            test_merged = []
            test_centroids = []
            clients_id = []
            nodes_id = []

            # create a dataframe with all the centroids and their informations
            for i in range(n_clients): 
                if (i!=max_index): # iterating an all the OTHER clients
                        test_graph = G[i]
                        test_nodes_i = [(node, data['CentroidValues']) for node, data in test_graph.nodes(data=True) if data.get('Layer') == l]
                        nodes_id.append(list(dict(test_nodes_i).keys()))
                        test_centroids_i = list(dict(test_nodes_i).values())
                        test_centroids.append(test_centroids_i)
                        test_merged_i = [False] * len(test_centroids_i)
                        clients_id_i = [i] * len(test_centroids_i)
                        clients_id.append(clients_id_i)

            flattened_centroids = [value for sublist in test_centroids for value in sublist]
            flattened_clients_id = [value for sublist in clients_id for value in sublist]
            flattened_nodes_id = [value for sublist in nodes_id for value in sublist]

            df_others = pd.DataFrame(list(zip(flattened_centroids,flattened_clients_id,flattened_nodes_id)),columns=['Centroid','ClientID','NodesID'])
            df_others['Distances'] = ''
            df_others['Merged'] = ''
            df_others['MergeWith'] = ''
            new_names = []
            distances = []
            merged = []
            mergwith = []
            max_dist = []

            for index, row in df_others.iterrows(): # for each node of the OTHER clients
                # calculate euclidean distances between the node and all the reference ones
                distances_i = [self.euclidean_distance(row['Centroid'], arr) for arr in ref_centroids]
                #print(f'distances_i {distances_i}' )
                distances.append(distances_i)
                max_dist_i = max(distances_i)
                max_dist.append(max_dist_i)

            # Normalize distances
            # Flatten the vector of distances
            #print(f'distances {distances}')
            flattened_vector = np.concatenate(distances)
            # Find the maximum and minimum values
            max_value = np.max(flattened_vector)
            min_value = np.min(flattened_vector)
            normalized_vector = np.array(self.normalize_vector(flattened_vector,min_value,max_value))
            distances = normalized_vector.reshape(-1, len(ref_centroids)).tolist()
            normalized_max = self.normalize_vector(max_dist,min_value,max_value)
            df_others['Distances'] = distances
            df_others['MaxDist'] = normalized_max

            for index, row in df_others.iterrows():
                # check if the node is mergeable, based on the threshold
                distances_i = row['Distances']
                merged_i = any(distance <= threshold for distance in distances_i)
                merged.append(merged_i)

                if(merged_i == True):
                    mergwith_i = np.argmin(distances_i)
                    new_name_i = f'layer{l} {mergwith_i}' # marked if mergeable
                else:
                    mergwith_i = ""
                    new_name_i = ""

                mergwith.append(mergwith_i)
                new_names.append(new_name_i)


            #df_others['Distances'] = distances
            df_others['Merged'] = merged   
            df_others['MergeWith'] = mergwith
            #df_others['MaxDist'] = max_dist
            df_others['NewName'] = new_names

            num_not_merged = (df_others['Merged'] == False).sum()
            num_nodes_add = my_max - min_generated # how many nodes I can still add

            if num_not_merged <= num_nodes_add: # if it is possible to add all the not-merged nodes as new reference ones
                df_others['NewNode'] = df_others['Merged'] == False # add all the not-merged nodes as new reference ones
                df_others['Merged'] = True # Solved for all the nodes

                # generating new names for the new reference ones
                unique_count = min_generated
                for index, row in df_others[df_others['NewNode'] == True].iterrows():
                    df_others.at[index, 'NewName'] = f'layer{l} {unique_count}'
                    unique_count += 1

            else: # only num_nodes_add can be actually added 
                # selecting the num_nodes_add most distant nodes as new nodes to add
                least_dist = df_others.nsmallest(num_nodes_add,'MaxDist')
                df_others['NewNode'] = False
                df_others.loc[least_dist.index,'NewNode'] = True 
                df_others.loc[least_dist.index,'Merged'] = True # Solved for these nodes

                # generating new names for the new reference ones
                for i, index in enumerate(least_dist.index):
                    df_others.at[index, 'NewName'] = f'layer{l} {i + min_generated}'

            # check if there are any other not-merged nodes
            if any(df_others['Merged'] == False):

                # extend the list of reference nodes (new reference nodes + old reference ones)
                new_nodes = df_others[df_others['NewNode'] == True] 
                new_centroids = new_nodes['Centroid'].tolist()
                new_ref_centroids = ref_centroids.copy()
                new_ref_centroids.extend(new_centroids) 

                # for each not-merged node
                for index, row in df_others.iterrows():
                    if row['Merged'] == False: # here normalization is not needed, the merging is not based on a threshold
                        # calculate the distance
                        new_dist = [self.euclidean_distance(row['Centroid'], arr) for arr in new_ref_centroids]
                        df_others.at[index,'Distances'] = new_dist
                        mergwith_i = np.argmin(new_dist)
                        df_others.at[index,'MergeWith'] = mergwith_i
                        df_others.at[index,'Merged'] = True
                        df_others.at[index,'NewName'] = f'layer{l} {mergwith_i}'

            # update node names for reference client
            old_names = [ref[0] for ref in ref_nodes]
            update_names = [f'layer{l} {i}' for i in range(len(old_names))]
            node_mapping = dict(zip(old_names, update_names))
            #nx.relabel_nodes(G[max_index], node_mapping, copy=False)
            G[max_index] = nx.relabel_nodes(G[max_index], node_mapping, copy=True)
            # update node names for all the other clients
            for i in range(n_clients): 
                if (i!=max_index): # iterating an all the OTHER clients
                        df_temp = df_others[df_others['ClientID']==i]
                        node_mapping = dict(zip(df_temp['NodesID'], df_temp['NewName']))
                        G[i] = nx.relabel_nodes(G[i], node_mapping, copy=True)

        G_new = G
        # end of the loop -> already worked on all the layers
        # calculate centroids value            
        new_centroids_dict = self.update_centroids(G)
        #print('centroid dict')
        #print(new_centroids_dict)
        # aggregate client graphs into server graph
        server_graph = nx.compose_all(G)
        # update centroid values of server graph
        nx.set_node_attributes(server_graph, new_centroids_dict, 'CentroidValues')

        layers = [0,1,2]
        n = 3
        final_cents = []
        for l in layers:
            nod_layer = [(node, data['CentroidValues']) for node, data in server_graph.nodes(data=True) if data.get('Layer') == l]
            my_dict = dict(nod_layer)
            m = len(my_dict)
            layer_cents = [[] for i in range(m)]
            for name, centroid in my_dict.items():
                index2 = int(name.split()[1])
                layer_cents[index2] = centroid
            final_cents.append(layer_cents)

        #print(f'final centroids {final_cents}')

        server_graph_without = server_graph.copy()
        for node in server_graph_without:
            server_graph_without.nodes[node].pop('Layer')
            server_graph_without.nodes[node].pop('Centroid')
            server_graph_without.nodes[node].pop('CentroidValues')


        return server_graph, server_graph_without, final_cents

