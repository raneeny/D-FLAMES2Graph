from Data_Preprocessing import ReadData
from ConvNet_Model import ConvNet
import numpy as np
import tensorflow.keras as keras
import sys, getopt
from High_Activated_Filters import HighlyActivated
import pandas
from itertools import *
from  functools import *
from Clustering import Clustering
from matplotlib import pyplot
import tensorflow.keras as keras
from matplotlib import pyplot
import numpy as np
from scipy import stats, integrate
from scipy.interpolate import interp1d
from scipy.spatial import distance
import matplotlib.pyplot as plt
import tensorflow as tf
from random import seed, shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from Embading import Graph_embading
from CustomGmeans import CustomGmeans
from ServerGraphAggregationNew import ServerGraphAggregation
import pickle
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from numpy import mean
from numpy import std
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,precision_score

import networkx as nx
import math
import time
import os



np.random.seed(0)
#import deepwalk
def readData(data_name,dir_name):
    dir_path = dir_name + data_name+'/'
    dataset_path = dir_path + data_name +'.mat'
    
    ##read data and process it
    prepare_data = ReadData()
    if(data_name == "HAR"):
        dataset_path = dir_name + data_name +'/train.pt'
        x_training = torch.load(dataset_path)
        x_train = x_training['samples']
        y_train = x_training['labels']
        dataset_path = dir_name + data_name +'/train.pt'
        x_testing = torch.load(dataset_path)
        x_test = x_testing['samples']
        y_test = x_testing['labels']
        x_train = x_train.cpu().detach().numpy()
        y_train = y_train.cpu().detach().numpy()
        x_test = x_test.cpu().detach().numpy()
        y_test = y_test.cpu().detach().numpy()
        #reshape array(num_sample,ts_len,dim)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[2], x_test.shape[1])
    elif(data_name == "PAMAP2"):
        dataset_path = dir_name + data_name +'/PTdict_list.npy'
        x_data = np.load(dataset_path)
        dataset_path = dir_name + data_name +'/arr_outcomes.npy'
        y_data = np.load(dataset_path)
        split_len = int(len(x_data)*0.9)
        x_train,x_test  = x_data[:split_len,:], x_data[split_len:,:]
        y_train,y_test  = y_data[:split_len,:], y_data[split_len:,:]
        
    else:
        prepare_data.data_preparation(dataset_path, dir_path)
        x_train = np.load(dir_path + 'x_train.npy')
        y_train = np.load(dir_path + 'y_train.npy')
        x_test = np.load(dir_path + 'x_test.npy')
        y_test = np.load(dir_path + 'y_test.npy')
    
    nb_classes = prepare_data.num_classes(y_train,y_test)
    y_train, y_test, y_true = prepare_data.on_hot_encode(y_train,y_test)
    x_train, x_test, input_shape = prepare_data.reshape_x(x_train,x_test)
    x_training = x_train
    y_training = y_train
 
    x_new1 = np.concatenate((x_train, x_test), axis=0)
    y_new1 = np.concatenate((y_train, y_test), axis=0)
    x_training, x_validation, y_training, y_validation = train_test_split(x_new1, y_new1, test_size=0.40,shuffle=True)
    x_validation,x_test,y_validation,y_test = train_test_split(x_validation, y_validation, test_size=0.50,shuffle=True)
    print(x_training.shape)
    print(x_validation.shape)
    print(x_test.shape)
    return x_training, x_validation, x_test, y_training, y_validation, y_true,y_test, input_shape,nb_classes

def trainModel(x_training, x_validation, y_training, y_validation,input_shape, nb_classes,epoch,weight,flag):
    ##train the model
    train_model = ConvNet()
    model = train_model.network_fcN(input_shape,nb_classes,weight,flag)
    train_model.trainNet(model,x_training,y_training,x_validation,y_validation,16,epoch)
    return model,train_model


def create_clients(data, label, num_clients, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as 
                data shards - tuple of images and label lists.
        args: 
            data_list: a list of numpy arrays of training data
            label_list:a list of binarized labels for each data
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1 
            
    '''

    #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]

    #randomize the data
    data = list(zip(data, label))
    random.shuffle(data)

    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    #number of clients must equal number of shards
    assert(len(shards) == len(client_names))

    return {client_names[i] : shards[i] for i in range(len(client_names))} 

def merge_traning_testing_data(x_training, x_validation, y_training, y_validation):
    #return data as data, label
    #data = np.concatenate((x_training, x_validation), axis=0)
    #data = np.concatenate((data,x_test), axis=0)
    #label = np.concatenate((y_training, y_validation), axis=0)
    #label = np.concatenate((label, y_test), axis=0)
    data = np.concatenate((x_training,x_test), axis=0)
    label = np.concatenate((y_training, y_test), axis=0)
    return data,label

def downsample_to_proportion(rows, proportion=1):
        i = 0
        new_data = []
        new_data.append(rows[0])
        k = 0
        for i in (rows):
            if(k == proportion):
                new_data.append(i)
                k = 0
            k+=1
        return new_data 

def clusering_array(period_active,comm_round):
    cluser_data_pre_list = []
    filter_lists = [[] for i in range(3)]
    for i in range(len(period_active)):
        for j in range(len(period_active[i])):
            for k in range(len(period_active[i][j])):
                filter_lists[j].append(period_active[i][j][k])

    cluser_data_pre_list.append([x for x in filter_lists[0] if x])
    cluser_data_pre_list.append([x for x in filter_lists[1] if x])
    cluser_data_pre_list.append([x for x in filter_lists[2] if x])
    #print(len(cluser_data_pre_list[0]))
    #print(len(cluser_data_pre_list[1]))
    #print(len(cluser_data_pre_list[2]))
    
    #print(cluser_data_pre_list[0][:5])
    cluser_data_pre_list1 = []
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[0], 1000))
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[1], 1000))
    cluser_data_pre_list1.append(downsample_to_proportion(cluser_data_pre_list[2], 1000))
    #cluser_data_pre_list1 = np.array(cluser_data_pre_list1)

    print('Starting Gmeans')
    print('layer 0')
    gmeans_instance0 = CustomGmeans(cluser_data_pre_list1[0],random_state=76,ccore=False,repeat=2).process()
    clusters0 = gmeans_instance0.get_clusters()
    centers0 = gmeans_instance0.get_centers()
    #print(centers0)
    print('layer 1')
    gmeans_instance1 = CustomGmeans(cluser_data_pre_list1[1],random_state=76,ccore=False,repeat=2).process()
    clusters1 = gmeans_instance1.get_clusters()
    centers1 = gmeans_instance1.get_centers()
    #print(centers1)
    print('layer 2')
    gmeans_instance2 = CustomGmeans(cluser_data_pre_list1[2],random_state=76,ccore=False,repeat=2).process()
    clusters2 = gmeans_instance2.get_clusters()
    centers2 = gmeans_instance2.get_centers()
    #print(centers2)

    cluster_centers = []
    cluster_centers.append(centers0)
    cluster_centers.append(centers1)
    cluster_centers.append(centers2)
    #np.save("cluster_central",cluster_central)
    print("saving the cluster")
    return cluster_centers

def normilization(data):
        i = 0
        datt = []
        maxi = max(data)
        mini = abs(min(data))
        while (i< len(data)):
            
            if(data[i] >=0):
                val = data[i]/maxi
            else:
                val = data[i]/mini
         
            datt.append(val)
            i += 1
            
        return datt

def fitted_cluster(data,cluster):
    data = data[0]
    data = normilization(data)
    cluster[0] = normilization(cluster[0])
    #print(data)
    #print(cluster[0])
    data = np.nan_to_num(data)
    cluster[0] = np.nan_to_num(cluster[0])
    #data = data[0]
    #print(data)
    #print(cluster[0][0])
    mini = distance.euclidean(data,cluster[0][0])
    cluster_id = 0
    count = 0
    for i in (cluster):
        clu_nor = normilization(i)
        data = np.nan_to_num(data)
        clu_nor = np.nan_to_num(clu_nor)
        dist = distance.euclidean(data,clu_nor[0])
        #print(dist)
        if(dist < mini):
            cluster_id = count
            mini = dist
        count+=1
            
    return cluster_id

def similarity_array(x,y):
    sim = [0]*len(x)
    cp_x = np.copy(x)

    for j in range(len(y)):
        #print(y[j])
        #print(cp_x)
        index_sim = fitted_cluster(y[j],cp_x)
        for i in range(len(x)):
            if(np.array_equal(cp_x[index_sim],x[i])):    
                sim[j] = i
                break
        cp_x = np.delete(cp_x, index_sim, axis=0)

    return sim

def similirity_node_name(centriods):
    i = 1
    clients_sim = []
    #for first client make it default
    cl_simi = []
    #shuffle each time centriod index
    ind_cent=random.randint(0, len(centriods)-1)
    new_cent = []
    new_cent.append(centriods[ind_cent])
    for j in range(len(centriods)):
        if(j != ind_cent):
            new_cent.append(centriods[j])
    centriods = new_cent
    for k in range (len(centriods[0])):
        simi_client = []
        for l in range(len(centriods[0][k])):
            simi_client.append(l)
        cl_simi.append(simi_client)
    clients_sim.append(cl_simi)
    #loop through clients centriod list
    while(i < len(centriods)):
        #loop through layers
        simi_client = []
        for k in range (len(centriods[0])):
            layers = centriods[0][k]
            simi_client.append(similarity_array(layers,centriods[i][k]))
        clients_sim.append(simi_client)
        i +=1
    #print(clients_sim)
    return clients_sim,ind_cent

def relabiling_client_graph(graph,maping):  
    mapping_graph = {}
    node_name = list(graph.nodes)
    for i in node_name:
        new_name = maping[int(i[5])][int(i[7])]
        new_name = 'layer%s %s' %(i[5],new_name)
        mapping_graph[i] = new_name
    G = nx.relabel_nodes(graph, mapping_graph)
    return G

def train_client_node(x_train,y_train,global_graph,centriod,flag,comm_round,client_name,nb_classes):
    ''' return: cluster these HAP and get the centriod
                global updated graph.
        args: 
            x_train: a training data
            y_train: training labels
            global_graph: global model from server
            comm_round: communication round
            client_name: name or data of the current client            
    '''
    #global_graph_ = []
    #global_graph_.append(global_graph)
    global_graph_ = global_graph.copy()
    # centriod_ = []
    #print(f'global_graph_ {global_graph_}')
    # centriod_.append(centriod)
    centriod_ = centriod.copy()
    #print(f'centriod_ {centriod}')
    #the client train node will first train a CNN based network
    #then extract the highly activated period(HAP) from the activated CNN nodes
    #build the graph based on the time relation between the HAP
    #x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.10,shuffle=True)
    weight = []
    flag = False
    print(comm_round)      
    if(comm_round > 0):
        #load_save model weight
        weight = keras.models.load_model(client_name)
        #if(comm_round == 6 or comm_round == 11 or comm_round ==19 ):
        #flag = False
        #else:
        flag = True
        model,train_model = trainModel(x_train, x_test, y_train, y_test,input_shape, nb_classes,3,weight.weights,flag)
    else:
        start_time = time.time()
        model,train_model = trainModel(x_train, x_test, y_train, y_test,input_shape, nb_classes,3,weight,flag)
        print("Tranin CNN time: %s seconds ---" % (time.time() - start_time))
    #save model weight
    model.save(client_name)
    
    #extract the MHAP
    if(comm_round == 0):
        start_time = time.time()
    visulization_traning = HighlyActivated(model,train_model,x_train,x_train,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_active,threshold = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
    if(comm_round == 0):
        print("Extract MHAP time: %s seconds ---" % (time.time() - start_time))
    #clustering the extracted MHAP
    if(comm_round == 0):
        start_time = time.time()
    cluster_central = clusering_array(period_active,comm_round)
    #print(cluster_central)
    if(comm_round == 0):
        print("Clustering time: %s seconds ---" % (time.time() - start_time))
    print('done_cluster')
    #build the graph
    G_w,sample_cluster_mhap = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],cluster_central,global_graph,threshold,6,10)
    #print('segment')
    ###here cheack if we have global graph, then update the current graph
    if(flag):
        global_graph_.append(G_w)
        #print(f'length before {len(centriod_)}')
        #print(len(centriod))
        centriod_.append(cluster_central)
        #print(f'length after {len(centriod_)}')
        #print(f'centriod_ {centriod}')
        #print(len(global_graph))
        #print(len(centriod))
        sga = ServerGraphAggregation()
        threshold = 0.75
        #print(f'global graph with second append: {global_graph_}')
        #G, G_w, cluster_central = sga.aggregate_graphs(global_graph,centriod,threshold)
        G, G_w, cluster_central = sga.aggregate_graphs_server(global_graph_,centriod_,threshold)
        
    name_sample = '%s_sample_cluster_mhap.npy' %(client_name)
    #np.save(name_sample,sample_cluster_mhap)
    #print('save_segment')
    #name_y = '%s_y.npy' %(client_name)
    #y_train = np.argmax(y_train, axis=1)
    #np.save(name_y,y_train)
    return cluster_central,G_w,model,train_model

#define function of Time Series Embedding
def timeseries_embedding(embedding_graph,node_names,timesereis_MHAP,number_seg):
    feature_list = []
    embed_vector = embedding_graph.wv[node_names]
    for i,data in enumerate(timesereis_MHAP):
        #compare the name with word_list and take its embedding
        #loop through segmant
        segmant = [[] for i in range(number_seg)]
        #print(len(data))
        for m,seg in enumerate(data):
            temp = [0 for i in range(len(embed_vector[0]))]
            #each seg has mhaps
            for k,mhap in enumerate(seg):
                for j,node in enumerate(node_names):
                    if(mhap == node):
                        temp += embed_vector[j]
                        break
            segmant[m].append(list(temp))
        feature_list.append(segmant)
    return feature_list

def gaussian_noise(x,mu,std):
    mu=mu
    std = std
    x = np.array(x)
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 


def test_clients(G,com_round,client_name,x_test,y_test,model,train_model,cluster_central):
    graph_embaded = Graph_embading(G)
    #graph_embaded.drwa_graph()
    node_names = graph_embaded.get_node_list()
    walks_nodes = graph_embaded.randome_walk_nodes(node_names)
    embaded_graph = graph_embaded.embed_graph(walks_nodes,com_round)
    #graph_embaded.plot_embaded_graph(embaded_graph,node_names)
    #add here new mhap for xtest
    x_test = gaussian_noise(x_test,0,5)
    visulization_traning = HighlyActivated(model,train_model,x_test,y_test,nb_classes,netLayers=3)
    activation_layers = visulization_traning.Activated_filters(example_id=1)
    period_active,threshold = visulization_traning.get_index_MHAP(activation_layers,kernal_size=[8,5,3])
    if(com_round == 0):
        start_time = time.time()
    g_l,sample_cluster_mhap_test = visulization_traning.get_graph_MHAP(activation_layers,[8,40,120],cluster_central,global_graph,threshold,6,10)
    if(com_round == 0):
        print("build_graph time: %s seconds ---" % (time.time() - start_time))
    #here merge all the data and then split
    y_test = np.argmax(y_test, axis=1)
    #sample_cluster_mhap =0
    #y_true = 0
    #name_sample = '%s_sample_cluster_mhap.npy'%(client_name)
    #sample_cluster_mhap = np.load(name_sample,allow_pickle=True)
    #name_y = '%s_y.npy' %(client_names[0])
    #y_true = np.load(name_y,allow_pickle=True)
    #for i in range(1,len(client_names)):
    #    name_sample1 = '%s_sample_cluster_mhap.npy'%(client_names[i])
    #    sample_cluster_cli1 = np.load(name_sample,allow_pickle=True)
    #    name_y1 = '%s_y.npy' %(client_names[i])
    #    y_true1 = np.load(name_y,allow_pickle=True)
        ##merge array 
    #    sample_cluster_mhap = np.concatenate((sample_cluster_mhap, sample_cluster_cli1), axis=0)
    #    y_true = np.concatenate((y_true, y_true1), axis=0)
    
    ##merge test and train
    #sample_cluster_mhap = np.concatenate((sample_cluster_mhap, sample_cluster_mhap_test), axis=0)
    #y_true = np.concatenate((y_true, y_test), axis=0)
    sample_cluster_mhap = sample_cluster_mhap_test
    y_true = y_test
    new_feature = timeseries_embedding(embaded_graph,node_names,sample_cluster_mhap,6)
    x_train_feature = [] 

    for m,data in enumerate (new_feature):
        segmant = []
        for j,seg in enumerate(data):
            segmant.append(seg[0])
        x_train_feature.append(segmant)

    x_train_new = []
    for i, data in enumerate (x_train_feature):
        seg = []
        for j in (data):
            for k in j:
                seg.append(k)
        x_train_new.append(seg)
    y_train =y_true
    X_train, X_test, y_train, y_test = train_test_split(x_train_new, y_true, test_size=0.10)
    ##adding noise to the x_test
    X_test = gaussian_noise(X_test,0,5)
    model = xgb.XGBClassifier()
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    #print(len(x_train_new))
    #print(len(y_true))
    n_scores = cross_val_score(model, x_train_new, y_true, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    #print('Client %s ,Communication round %s, Accuracy: %.3f (%.3f)' % (client_name,com_round,mean(n_scores), std(n_scores)))
    if(com_round == 0):
        start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if(com_round == 0):
        print("XGBoost time: %s seconds ---" % (time.time() - start_time))
    balance_acc = balanced_accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred, average='weighted')
    pres_val = precision_score(y_test, y_pred,average='weighted')
    #print('Client %s ,Communication round %s, Accuracy: %.3f' % (client_name,com_round,accuracy))
    #return mean(n_scores),TN,FN,TP,FP
    #return accuracy,balance_acc,f1_sc,pres_val
    return mean(n_scores),balance_acc,f1_sc,pres_val

experiment = 'EXPERIMENT25'
repetitions = 3
for r in range(repetitions):
    data_name = 'ECG'
    dir_name = '../data/'
    x_training, x_validation, x_test, y_training, y_validation,y_true,y_test,input_shape, nb_classes = readData(data_name,dir_name)
    #data,label = merge_traning_testing_data(x_training, x_validation, y_training, y_validation)
    initials = f'{data_name}_{experiment}_rep{r}_clients'
    clients = create_clients(x_training,y_training,4,initial=initials)

    common_round = 100
    acc_clients = [[] for i in range(len(clients))] 
    flag= False
    global_graph = []
    contriod_0 = []
    BalanceACC_= [[] for i in range(len(clients))]
    F1_=[[] for i in range(len(clients))]
    Precision_=[[] for i in range(len(clients))]
    Flag_=[0 for i in range(len(clients))]

    ##averge values
    acc_avg = [] 
    avg_BalanceACC_= []
    avg_F1_=[]
    avg_Precision_=[]
    g_graphs = []
    g_cent = []
    save_graph = True
    for i in range(common_round):
        global_graphs = []
        centriod =[]
        client_names= list(clients.keys())
        #random.shuffle(client_names)
        count=0
        for client in client_names:
            #here we should get the traning data from clients
            x_train = []
            y_train = []
            for(X_test, Y_test) in clients[client]:
                x_train.append(X_test)
                y_train.append(Y_test)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            #train each client model for n rounds
            #if(count <= 1):
            if(i >0):
                flag = True
            if(i == 0):
                start_time = time.time()
            client_centriod,client_global_graph,model,train_model = train_client_node(x_train,y_train,global_graph,contriod_0,flag,i,client,nb_classes)
            if(i == 0):
                print("traning the clients time: %s seconds ---" % (time.time() - start_time))
            #nx.draw(client_global_graph)
            global_graphs.append(client_global_graph)
            centriod.append(client_centriod)
            count +=1
            
        #update the global model
        if(i == 0):
            start_time = time.time()
        threshold = 0.75
        sga = ServerGraphAggregation()
        global_g, global_g_w,contriod_c = sga.aggregate_graphs_clients(global_graphs,centriod,threshold)
        global_graph = []
        contriod_0 = []
        
        folder_path = f'../data/{experiment}graphs_rep{r}/'

        # Check if the folder exists, if not, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        path_graph = f'{folder_path}{data_name}_server_graph_round{i}.gpickle' 
        #saving graph
        with open(path_graph, 'wb') as file_graph:
            pickle.dump(global_g, file_graph)
        path_graphW = f'{folder_path}{data_name}_server_graphW_round{i}.gpickle'
        with open(path_graphW, 'wb') as file_graph:
            pickle.dump(global_g_w, file_graph) 
        global_graph.append(global_g_w)
        contriod_0.append(contriod_c)
        """     if(save_graph):
            global_graph = []
            contriod_0 = []
            global_graph.append(global_g)
            contriod_0.append(contriod_c)
        else:
            global_g=global_graph[0]
            contriod_c = contriod_0[0] """
        if(i == 0):
            print("Graph aggregation time: %s seconds ---" % (time.time() - start_time))
        #send the client the new graph, re do maping in way suits client node
        #do testing by embed the graph and see the acc on test data
        c=0
        #do global testing

        for client in client_names:
            if(i == 0 and c == 0):
                start_time = time.time()
            accur,balance_acc,f1_sc,pres_val=test_clients(global_g,i,client,x_test,y_test,model,train_model,contriod_c)
            if(i == 0 and c == 0):
                print("Test client time: %s seconds ---" % (time.time() - start_time))
            acc_clients[c].append(accur)
            BalanceACC_[c].append(balance_acc)
            Precision_[c].append(pres_val)
            F1_[c].append(f1_sc)
            c+=1
        
        #get avergae of alll client and save to array
        #flage use current graph = true
        if(i==0):
            avg_acc = 0
            bal_a=0
            av_f1=0
            av_pre=0
            for c in range(len(client_names)):
                avg_acc+=acc_clients[c][i]
                bal_a+=BalanceACC_[c][i]
                av_f1+=F1_[c][i]
                av_pre+=Precision_[c][i]
            acc_avg.append(avg_acc/len(client_names))
            avg_BalanceACC_.append(bal_a/len(client_names))
            avg_F1_.append(av_f1/len(client_names))
            avg_Precision_.append(av_pre/len(client_names))
            save_graph = True
        elif(i >0):
            avg_acc = 0
            bal_a=0
            av_f1=0
            av_pre=0              
            for c in acc_clients:
                avg_acc+=c[i]
            avg_acc = avg_acc/len(client_names)
            if(0<(acc_avg[i-1]-avg_acc) <= 0.03 or (avg_acc>acc_avg[i-1])):
                acc_avg.append(avg_acc)
                for c in range(len(client_names)):
                    bal_a+=BalanceACC_[c][i]
                    av_f1+=F1_[c][i]
                    av_pre+=Precision_[c][i]
               
                avg_BalanceACC_.append(bal_a/len(client_names))
                avg_F1_.append(av_f1/len(client_names))
                avg_Precision_.append(av_pre/len(client_names))
                save_graph = True
            else:
                save_graph = False
                acc_avg.append(acc_avg[i-1])
                avg_BalanceACC_.append(avg_BalanceACC_[i-1])
                avg_F1_.append(avg_F1_[i-1])
                avg_Precision_.append(avg_Precision_[i-1])
                save_graph = True
        print('Communication round %s, Average Accuracy: %.3f' % (i,acc_avg[i]))
        print('Communication round %s, Average BalanceAccuracy: %.3f' % (i,avg_BalanceACC_[i]))
        print('Communication round %s, Average F1: %.3f' % (i,avg_F1_[i]))
        print('Communication round %s, Average Precision: %.3f' % (i,avg_Precision_[i]))

    df = pd.DataFrame({'Accuracy': acc_avg, 'BalanceACC' : avg_BalanceACC_, 'Precision' : avg_Precision_, 'F1' : avg_F1_})
    folder_path = f'../data/{experiment}results_rep{r}/'

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    name = f'{folder_path}{data_name}_clients_graph_accuracy.xlsx'
    if os.path.exists(name):
        os.remove(name)
    df.to_excel(name)



    for i in range(len(acc_clients)):
        save_arr = []
        name = f'{folder_path}{data_name}_client_{i}_graph_accuracy.xlsx'
        if os.path.exists(name):
            os.remove(name)
        df = pd.DataFrame({'Accuracy': acc_clients[i], 'BalanceACC' : BalanceACC_[i], 'Precision' : Precision_[i], 'F1' : F1_[i]})
        df.to_excel(name)