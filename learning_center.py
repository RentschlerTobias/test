from oslo_concurrency import lockutils
class learning_center():
    PREDICTION_NET = None
    CLUSTER_IDS_BAD = None
    CLUSTER_IDS_VALUED = None
    PATH_RUN_DATA = None
    EVAL_COUNTER = 0
    CLUSTER_IDS_ASSIGNED = None
    CASE_DATA = None


    import torch
    import logging
    import os
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import pygmo as pg
    import numpy as np
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
    from pyDtOO import dtClusteredSingletonState as stateCounter
    ###IMPORT KI MODELS###
    from scripts.models import auto_encoder_convolution_medium
    import importlib
    from matplotlib import pyplot as plt
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import NearestNeighbors
    from scripts.tistosPyBib import tistosRunner
    import copy

    ### FOR DEBUGGING REASONS - REIMPORT KI MODELS ###
    import sys 
    importlib.reload(sys.modules['scripts.models'])
    from scripts.models import auto_encoder_convolution_medium

    
    def __init__(self, runData = None, cluster_anzahl = None):
        self.PREFIX = 'T1'
        self.cl_alg = 'SpectralClustering'
        self.model_c = [] ### MODEL FOR UNSUPERVISED CLUSTERING OF FLOW FIELD
        if cluster_anzahl == None:
            self.anzahl_cluster = 20
        else:
            self.anzahl_cluster = cluster_anzahl
        self.cluster_ids = [] 
        self.cluster_ids_assigned = [] #[case_id,cluster_id]
        self.case_data = None
        self.T = [] #tensors [area,[id,tensors]] ----> indices: [area_id][1][:] <--- all tensors of area_id
        self.z = [] #latent space same structures as tensors but with corresponding latent space z
        self.area_names = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']

        if runData == None:
            self.pathRunData = './runData_1000'
        else:
            self.pathRunData = runData
        self.pathFlowField = self.pathRunData + '/flowfield_hexa/'
        
        self.logger_cluster = self.logging.getLogger('cluster_logger')
        self.logger_cluster.setLevel(self.logging.DEBUG)
        self.file_handler =self.logging.FileHandler('cluster.log')
        formatter = self.logging.Formatter('%(asctime)s %(levelname)s : %(message)s')
        self.file_handler.setFormatter(formatter)
        self.logger_cluster.addHandler(self.file_handler)
        self.DOF = self.tistosRunner.DoF()
        self.op = '_all_' # considering all load conditions
        if learning_center.PREDICTION_NET == None:
            print("First Object: Retrain first:")
            self.retrain(self.anzahl_cluster, max_iter = 10, train = False)


    ### LOADING DATA... ###
    def set_device(self):
        ### SELECT DEVICE ###
        device = (
            "cuda"
            if self.torch.cuda.is_available()
            else "mps"
            if self.torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        #self.logger_cluster.info(f"Using {device} device")
        return device


    def initialize(self):
        # Define the folder name
        runData = './runData/'
        flowfield_hexa = 'flowfield_hexa'
        # Create the directory
        for directory in [runData, runData + flowfield_hexa]:
            if not self.os.path.exists(directory):
                self.os.makedirs(directory)
        
    
    def set_cluster_anzahl(self, anzahl):
        self.anzahl_cluster = anzahl
        print(f'Cluster_Anzahl set to {self.anzahl_cluster}.')

    
    def load_model(self, model_type = "clustering", pretrained=True):
        print(f"*** Loading Model {model_type} ***")
        for i in range(len(self.area_names)):
            padding_x = self.get_padding(self.T[i+1][0], 'x', 5)
            padding_y = self.get_padding(self.T[i+1][0], 'y', 5)
            padding_z = self.get_padding(self.T[i+1][0], 'z', 5)
            self.model_c.append(self.auto_encoder_convolution_medium(padding_x, padding_y, padding_z))
            if pretrained:
                try:
                    self.model_c[-1].load_state_dict(self.torch.load(f'model_trained/model_trained_{self.area_names[i]}.pth'))
                    print(f'Pretrained: Loaded model_trained_{self.area_names[i]}.pth')
                except Exception as e:
                    print("Couldn't find matching model-data!")
                    print(e)
                    print("If you're not using a cuda device. Delete pretrained models and retrain yourself.")
                    #self.logger_cluster.warning(f"Failed loading Trained AE: Error:")
                    #self.logger_cluster.warning(e)


    def load_data(self, number_min = 100):
        self.stateCounter.DATADIR = self.pathRunData
        #self.logger_cluster.info(f"StateCounter runData path: {self.stateCounter.DATADIR}.")
        print(f"StateCounter runData path: {self.stateCounter.DATADIR}.")
        if self.stateCounter.currentMaxId() == 0:
            pass
        else:
            self.max_id = self.stateCounter.currentMaxId()
            I,O,F = self.stateCounter.fullRead()
            M = self.stateCounter.fullAddRead(
                ['P', 'dH', 'eta', 'VCav', 'history','islandID'],
                [dict, dict, dict, dict, dict,int]
                )

            eta = []
            for i in range(M['eta'].shape[0]):
                eta_tl = M['eta'][i]['tl']
                eta_n = M['eta'][i]['n']
                eta_vl = M['eta'][i]['vl']
                eta.append([eta_tl, eta_n, eta_vl])
            eta = self.np.array(eta)

            dH = []
            for i in range(M['dH'].shape[0]):
                dH_tl = M['dH'][i]['tl']
                dH_n = M['dH'][i]['n']
                dH_vl = M['dH'][i]['vl']
                dH.append([dH_tl, dH_n, dH_vl])
            dH = self.np.array(dH)
               
            VCav = []
            for i in range(M['VCav'].shape[0]):
                VCav_tl = M['VCav'][i]['tl']
                VCav_n = M['VCav'][i]['n']
                VCav_vl = M['VCav'][i]['vl']
                VCav.append([VCav_tl, VCav_n, VCav_vl])
            VCav = self.np.array(VCav)

            P = []
            for i in range(M['P'].shape[0]):
                P_tl = M['P'][i]['tl']
                P_n = M['P'][i]['n']
                P_vl = M['P'][i]['vl']
                P.append([P_tl, P_n, P_vl])
            P = self.np.array(P)

            ### Single Objective: Mean ###
            F_mean = self.copy.deepcopy(F)
            F_mean = F_mean*(1/3)
            F_mean = self.np.sum(F_mean, axis = 1)
            self.case_data = [I,O,F, F_mean, eta, dH, VCav, P]
            learning_center.CASE_DATA = self.case_data


    def get_data(self, case):
        case_indice = case - 1
        print(f"Case-Data for {self.PREFIX}_{self.case_data[0][case_indice]}.")
        print("[Fmean,        F,           Objectives]")
        return [self.case_data[3][case_indice], self.case_data[2][case_indice], self.case_data[1][case_indice]]

        
    def normalize_tensors(self):
        tensor_list = [self.T[0]]         
        ### GET (MAXIMUMG) SCALING FACTOR PER CHANNEL ###
        scaling_factors = [self.np.inf,self.np.inf,self.np.inf,self.np.inf]
        for channel in range(4):
            for area in self.area_names:
                model_id = self.area_names.index(area)
                scaling_factor_new = 2/(self.np.quantile(self.T[model_id+1][:,channel,:,:,:].abs().numpy(), 0.95))
                if scaling_factor_new < scaling_factors[channel]:
                    scaling_factors[channel] = scaling_factor_new
            if scaling_factors[channel] == self.np.inf:
                print(f"Error: Didn't define Scaling Factor for Channel {channel}.")     
        print("Final Scaling Factors: Channels: [1,2,3,4]")
        print(scaling_factors)           
        for area in self.area_names:
            model_id = self.area_names.index(area)
            input_data = self.T[model_id+1]
            ### NORMALIZE DATA FOR EVERY CHANNEL ###
            for i in range(4):
                input_data[:,i,:,:,:] = (input_data[:,i,:,:,:]*scaling_factors[i]).tanh()
            tensor_list.append(input_data)
        return tensor_list


    def check_tensor_range(self):
        try:
            for i in range(len(self.area_names)):
                print(self.area_names[i])
                for c in range(4):
                    print(f'Channel {c}: ')
                    print("STD:", self.T[i+1][:,c,:,:,:].std())
                    print("MAX:", self.T[i+1][:,c,:,:,:].max())
                    print("MIN:", self.T[i+1][:,c,:,:,:].min())
                print('')  
        except:
            print("Failed: Load Tensors first.")
        

    @lockutils.synchronized('tensorIO', external = True)
    def load_tensors(self, normalize = True):
        self.load_data()
        tensor_ids_ok = [self.PREFIX + '_' + str(self.case_data[0][i])+self.op for i in range(len(self.case_data[0])) if (self.case_data[3][i] < 1)]
        self.tensor_ids_ok = tensor_ids_ok

        for area in self.area_names:
            tensor_id, tensor_list = self.get_tensor(area)
            tensor_list_new = [tensor_list[i] for i in range(len(tensor_list)) if tensor_id[i].split("b")[0] in tensor_ids_ok]
            tensor_id_new = [tensor_id[i] for i in range(len(tensor_list)) if tensor_id[i].split("b")[0] in tensor_ids_ok]
            self.tensor_id_new = tensor_id_new
            if area == 'b1':
                self.T = [tensor_id_new]
            input_data = [t.unsqueeze(0) for t in tensor_list_new]
            input_data = self.torch.cat(input_data, dim=0)
            self.T.append(input_data)
            print('Loaded Tensors from ' , area)

        if normalize:
            self.T_denormalized = self.copy.deepcopy(self.T)
            tensor_list = self.normalize_tensors()
            self.T = tensor_list #overwrite Tensor data with normalized tensors
            print("Normalized Tensors.")
            self.T_denormalized[0] = self.np.array([name.split("_b")[0] for name in self.T_denormalized[0]])
        self.T[0] = self.np.array([name.split("_b")[0] for name in self.T[0]])

            
    def get_tensor(self, area, directory = ''):
        import time
        start_time_sort = time.time()
        tensor_list = []
        tensor_id = []
        if directory == '':
            directory = self.pathFlowField
        filenames = self.os.listdir(directory)
        if area != 'complete':
            filenames = [f for f in filenames if self.os.path.isfile(self.os.path.join(directory, f)) and area in f and self.op in f]
            filecases = [int(s.split('_')[1]) for s in filenames]
            indices = sorted(range(len(filenames)), key = lambda k: filecases[k])
            filenames = [filenames[k] for k in indices]
            filecases = [filecases[k] for k in indices]
        end_time_sort = time.time()

        start_time_load = time.time()
        for file in filenames:
            tensor_list.append(self.torch.load(directory+file))
            tensor_id.append(file)
        end_time_load = time.time()
        
        # Calculate the time difference
        time_diff_sort = end_time_sort - start_time_sort
        time_diff_load = end_time_load - start_time_load
        
        # Print the time difference
        print(f"Time taken for sorting and loading File-List: {time_diff_sort} seconds")
        print(f"Time taken for loading: {time_diff_load} seconds")
        return self.np.array(tensor_id), tensor_list
    

    def get_latent_data(self):
        ### CHECK IF MODEL IS ALREADY LOADED ###
        if self.model_c == []:
            self.load_model(model_type = "clustering", pretrained=True)

        ### SELECT DEVICE ###
        #device =  self.set_device()
        device = 'cpu'
        self.z = [self.T[0]]
        for area in self.area_names:
            i = self.area_names.index(area)
            tensor = self.T[i+1].to(device)
            self.model_c[i] = self.model_c[i].to(device)
            z = self.model_c[i].encoder(tensor)
            #z = z.detach().cpu().numpy()
            z = z.detach().numpy()
            shape_z = z.shape
            z = z.reshape(shape_z[0], -1)
            self.z.append(z)
        
        self.T = None
        self.T_denormalized = None
        #self.logger_cluster.info("Tensor deleted.")
        print("Got Latent Space Data. Delete Tensors...")
    

    def get_padding(self, input_data, index, anzahl):
        #input_data = input_data[0]
        if index == 'x':
            dim = input_data.shape[-3]
        elif index == 'y':
            dim = input_data.shape[-2]
        elif index == 'z':
            dim = input_data.shape[-1]
    
        if dim %2 == 0:
            padding_4 = 1
            dim_1 = dim/2
        else:
            padding_4 = 0
            dim_1 = dim/2 + 0.5      
        if  dim_1 % 2 == 0:
            padding_3 = 1
            dim_2 = dim_1/2
        else:
            padding_3 = 0
            dim_2 = dim_1/2 + 0.5
        if dim_2 % 2 == 0:
            padding_2 = 1
            dim_3 = dim_2/2 
        else:
            padding_2 = 0
            dim_3 = dim_2/2 + 0.5
        if dim_3 % 2 == 0:
            padding_1 = 1
            dim_4 = dim_3/2
        else:
            padding_1 = 0
            dim_4 = dim_3/2 + 0.5  
        if dim_4 % 2 == 0:
            padding_0 = 1
        else:
            padding_0 = 0
        if anzahl == 2:
            return [padding_3, padding_4]
        elif anzahl == 3:
            return [padding_2, padding_3, padding_4]
        else:
            return [padding_0, padding_1, padding_2, padding_3, padding_4]


    
            
    ### TRAINING ###
    def train_model_c(self, area, max_iter = 1000):
        print(f"Train Model_C for area {area}")
        #self.logger_cluster.info(f"Train Model_C for area {area}")
        model_id = self.area_names.index(area)
        input_data = self.T[model_id+1]
 
        optimizer = self.torch.optim.Adam(self.model_c[model_id].parameters(), lr = 0.001, weight_decay = 1e-8)
        scheduler = self.torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
        MSE_loss = self.torch.nn.MSELoss()

        ### SELECT DEVICE ###
        device = (
            "cuda"
            if self.torch.cuda.is_available()
            else "mps"
            if self.torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device for Training.")
        #self.logger_cluster.debug(f"Using {device} device for Training.")

        ### DATASET CLASS ###
        class flow_dataset(self.Dataset):
            def __init__(self, input_data):
                self.data = input_data

            def __len__(self):
                return len(self.data)
        
            def __getitem__(self, idx):
                feature = self.data[idx]
                return feature
                
        dataset = flow_dataset(input_data) 
        length_dataset = input_data.shape[0]
        batch_size = 2
        length_train = int(length_dataset*0.65)
        length_val = length_dataset-length_train-1
        length_test = 1
        train_data, val_data,test_data = self.torch.utils.data.random_split(dataset, [length_train,length_val,length_test] )
        
        ### DEFINE DATALOADER FOR MANAGING DATASET IN TRAINING ###
        loader_train = self.DataLoader(train_data, batch_size = batch_size, shuffle=True) 
        loader_val = self.DataLoader(val_data, batch_size = batch_size, shuffle = True)
        loader_test = self.DataLoader(test_data, batch_size = batch_size, shuffle = True)
        
        ### SAVE LOSS VALUES ###
        best_loss = self.np.inf

        ### PRINT INITIAL LOSS ###
        loss = 0
        for i, data in enumerate(loader_train):
                self.model_c[model_id] = self.model_c[model_id].to(device) 
                self.model_c[model_id].eval()
                data = data.to(device)
                z = self.model_c[model_id].encoder(data)
                X_recon = self.model_c[model_id].decoder(z)
                loss += MSE_loss(X_recon,data)
        loss = loss.item()/(len(loader_train))
        print("Initial_Loss: ", loss)
        #self.logger_cluster.debug(f"Initial_Loss: {loss}.")
    
        ### TRAIN: ###
        print(f"Start Training for {area}")
        #self.logger_cluster.debug(f"Start Training for {area}")

        if event == None:
            class holder():
                def __init__(self):
                    self.parallel = False

                def is_set(self):
                    return False
            event = holder()

        for epoch in range(max_iter):
            train_loss = 0
            ### COMPUTE TRAINING LOSS ###
            for i, data in enumerate(loader_train):
                self.model_c[model_id] = self.model_c[model_id].to(device) 
                self.model_c[model_id].train()
                data = data.to(device)
                optimizer.zero_grad()
                z = self.model_c[model_id].encoder(data)
                X_recon = self.model_c[model_id].decoder(z)
                loss = MSE_loss(X_recon,data) #+ 0.001* self.model_c[model_id].l1_regularization()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() 
                
                ### COMPUTE VALIDATION LOSS ### 
                self.model_c[model_id].eval()
                loss_val = 0
                used_val = min(8,length_val)
                for _ in range(used_val):
                    val_data = next(iter(loader_val))
                    val_data = val_data.to(device)
                    val_z = self.model_c[model_id].encoder(val_data)
                    val_recon = self.model_c[model_id].decoder(val_z)
                    loss_val += MSE_loss(val_data, val_recon).item()
                loss_val = loss_val / used_val
                
                
                ### SAVE MODEL IF BEST LOSS ###
                if loss_val < best_loss:
                    best_loss = loss_val
                    ### save trained model: ###
                    self.torch.save(self.model_c[model_id].state_dict(), f'./model_trained/model_trained_{area}.pth')
    
            train_loss = train_loss / len(loader_train)
            ### PRINT OUTPUT ###
            if epoch % 10 == 0:
                #self.logger_cluster.info(f'{area}: Epoche: {epoch},  Train-Loss: {train_loss},  Validation-Loss: {loss_val}, Best-Validation-Loss: {best_loss}')
                print(f'{area}: Epoche: {epoch},  Train-Loss: {train_loss},  Validation-Loss: {loss_val}, Best-Validation-Loss: {best_loss}')

        print(f'Finished Training for Area {area}.')
        print(f'Best Validation-Loss: {best_loss}\n')
        #self.logger_cluster.debug(f'Epoche: {epoch},  Train-Loss: {train_loss},  Validation-Loss: {loss_val}, Best-Validation-Loss: {best_loss}')
        #self.logger_cluster.info(f'Finished Training for Area {area}: Best Validation-Loss: {best_loss}\n')
        
        

                
    def train_model_c_all(self, max_iter = 100):
        if self.T == None or self.T == []:
            #self.logger_cluster.info(f"Reload Tensors for Training.")
            print(f"Reload Tensors for Training.")
            self.load_tensors()
        for area in self.area_names:
            self.train_model_c(area, max_iter = max_iter)
    
        
    def get_cluster_ids(self, algo = 'KMeans', mode = 'all', retrain = True):
        #self.logger_cluster.info(f"Get Cluster IDs with {algo}.")
        print(f"Get Cluster IDs with {algo}.")
        self.cluster_ids = []
        anzahl_cluster = self.anzahl_cluster
        if retrain == True:
            self.cluster_algo_list = []
        for area in self.area_names:
            area_id = self.area_names.index(area)
            z = self.z[area_id+1]
            z_name = self.z[0]
                
            ### eliminate bad turbines ###
            if mode == 'well':
                if area_id == 0:
                    print("Only choose well simulated cases: --->")
                good_cases = [f'{self.PREFIX}_{i}{self.op[:-1]}' for i in self.eliminate_bad_fitness()]
                indices = self.np.where(self.np.isin(z_name,good_cases))[0]
                z_choose = z[indices]
                z_name = z_name[indices]
                if len(z) == len(z_choose):
                    print("Caution nothing was eliminated...!")

            elif mode == 'best':
                if area_id == 0:
                    print("Only cluster best simulations.")
                best_cases = [helper + self.op[:-1] for helper in self.case_lowest]
                indices = self.np.where(self.np.isin(z_name,best_cases))[0]
                z_choose = z[indices]
                z_name = z_name[indices]
            else:
                z_choose = z

            if retrain == False:
                if area_id == 0:
                    ### ONLY WORKS WITH KMeans...###
                    print("Not retraining cluster_algorithm. Set retrain = True for retraining")
                cluster_id_helper = self.cluster_algo_list[area_id].predict(z_choose)

            elif retrain == True:
                if algo == 'KMeans':
                    KMeans_ = self.KMeans(n_clusters=anzahl_cluster, random_state=42)
                    clustering = KMeans_.fit(z_choose)
                    self.cluster_algo_list.append(clustering)
                    cluster_id_helper = clustering.labels_
                
                elif algo == 'SpectralClustering':
                    SC = self.SpectralClustering(n_clusters=anzahl_cluster, assign_labels='kmeans', random_state=0)
                    clustering = SC.fit(z_choose)
                    self.cluster_algo_list.append(clustering)
                    cluster_id_helper = clustering.labels_

                else:
                    print(f'{algo} is not available')
            self.cluster_ids.append(cluster_id_helper)
        cluster_ids_assigned = self.np.stack(self.cluster_ids, axis=1)
        print(f'Shape of Cluster_IDS_Assigned: {cluster_ids_assigned.shape}')
        print("[Amount of Individuals, Amount of FlowAreas]")
        self.cluster_ids_assigned = [z_name, cluster_ids_assigned]
        learning_center.CLUSTER_IDS_ASSIGNED = self.cluster_ids_assigned

        self.value_cluster_ids()
    
    
    def value_cluster_ids(self):
        ### VALUE CLUSTER IDS ###
        self.cluster_ids_valued = self.np.unique(self.np.array(self.cluster_ids_assigned[1]), axis=0)
        eta_nl = []
        eta_nl_min = []
        fitness_values = []
        fitness_mean = []
        fitness_min = []
        fitness_div = []
        fitness_worst = []
        obj_deviation = []
        number_of_individuals = []
        eta_nl_deviation = []
        obj = []
        for cluster_id in self.cluster_ids_valued:
            fitness = self.give_fitness(cluster_id, output = 'off')[2]
            eta_nl_helper = self.give_fitness(cluster_id, output = 'off')[4][:,1]
            obj_list = self.give_fitness(cluster_id, output = 'off')[1]
            obj_array = self.np.stack(obj_list)
            obj_dev_element = self.np.std(obj_array, axis=0)
            
            eta_nl.append(self.np.sort(eta_nl_helper)) 
            eta_nl_min.append(self.np.min(eta_nl_helper))
            fitness_values.append(self.np.array(fitness)) 
            eta_nl_deviation.append(self.np.std(eta_nl_helper))
            obj_deviation.append(obj_dev_element)
            obj.append(obj_array)
            fitness_mean.append(self.np.mean(self.np.array(fitness)))
            fitness_min.append(self.np.min(self.np.mean(self.np.array(fitness), axis = 1)))
        self.cluster_ids_valued = [self.cluster_ids_valued, eta_nl ,eta_nl_min ,fitness_values, eta_nl_deviation, obj_deviation, obj, fitness_mean, fitness_min]
        
        ### Cluster_IDS_Bad
        fit_grenze = self.np.quantile(self.case_data[2][-250:,0][self.case_data[2][-250:,0] < 1.0], 0.40) #Median/Mittelwert of Fitness-ETA für not failed Sims
        fit_grenze_mean = self.np.mean(self.case_data[2][-250:,0][self.case_data[2][-250:,0] < 1.0])
        print(f"Fit-Grenze: Fitness Eta 40 Percent Quantile: {fit_grenze:.3f}.Fitness Eta Mean: {fit_grenze_mean:.3f}. Max_ID: {self.case_data[0][-1]}")
        self.cluster_ids_bad = [self.cluster_ids_valued[0][i] for i in range(len(self.cluster_ids_valued[0])) if self.np.mean(self.cluster_ids_valued[3][i][:,0]) > fit_grenze]
        self.cluster_ids_bad = self.np.array(self.cluster_ids_bad)
        learning_center.CLUSTER_IDS_BAD = self.cluster_ids_bad

        ### Sort Cluster_IDs  with F_Mean ###
        sorted_indices = sorted(range(len(self.cluster_ids_valued[-1])), key=lambda i: self.cluster_ids_valued[-1][i])
        for i in range(len(self.cluster_ids_valued)):
            self.cluster_ids_valued[i] = [self.cluster_ids_valued[i][indice] for indice in sorted_indices]
        learning_center.CLUSTER_IDS_VALUED = self.cluster_ids_valued
    

    def find_cases(self, cluster_id):
        index = self.np.where(self.np.all(learning_center.CLUSTER_IDS_ASSIGNED[1] == cluster_id, axis=1))[0]
        cases = [learning_center.CLUSTER_IDS_ASSIGNED[0][i] for i in index]
        return cases


    def give_fitness(self, cluster_id, output = 'on'):
        if output == 'on':
            print("[CaseIDs,   Objectives,   Fitness-Werte,   Fitness-Mittelwert, Eta, dH , VCav, P ]")
        case_id = self.find_cases(cluster_id)
        fitness = [[], [], [],[],[],[],[],[]]
        for i in case_id:
            i = int ( i.split(f'{self.PREFIX}_')[1].split(self.op[:-1])[0] )
            fitness[0].append(f'{self.PREFIX}_{i}'+self.op[:-1]) #Case_ID
            fitness[1].append(learning_center.CASE_DATA[1][i-1]) #Objectives
            fitness[2].append(learning_center.CASE_DATA[2][i-1]) #Fitness
            fitness[3].append(learning_center.CASE_DATA[3][i-1]) #Fitness-Mean
            fitness[4].append(learning_center.CASE_DATA[4][i-1]) ###Eta
            fitness[5].append(learning_center.CASE_DATA[5][i-1]) ###Fallhöhe 
            fitness[6].append(learning_center.CASE_DATA[6][i-1]) ###VCav
            fitness[7].append(learning_center.CASE_DATA[7][i-1]) #P
        for k in range(len(fitness)):
            fitness[k] = self.np.array(fitness[k])
        return fitness

    def give_cluster_id(self,casenumber, output = 'on'):
        if output == 'on':
            print(f"Cluster_id for case {self.PREFIX}_{casenumber}")
        case_list = [ int(s.split(f"{self.PREFIX}_")[1].split(self.op[:-1])[0]) for s in self.cluster_ids_assigned[0]]
        case_id = self.np.where(self.np.isin(case_list,casenumber))[0]
        return self.cluster_ids_assigned[1][case_id]

    def eliminate_bad_fitness(self):
        indices_ok = self.np.where(self.case_data[3] < 1.0)[0] ###eliminate failed fitness or "very" bad turbines
        ### Delete cases from initialization ###
        indices_ok = indices_ok[indices_ok >= 84] ####85 depends on islands and population per island from initialization dataset
        ### Indices +1 = CaseNumber ###
        cases_ok = indices_ok +1
        return cases_ok

    def extract_data(self, area):
        objectives = self.np.empty((0,30))
        targets = self.np.empty(0,)
        case_id = []
        for i, case in enumerate(self.case_data[0]):
           obj = self.np.array([self.case_data[1][i]])
           if self.give_cluster_id(case,output='off').shape[0] != 0: # only if cluster-id is known
               target_id = self.give_cluster_id(case,output='off')[0][area]
               targets = self.np.append(targets, self.np.array([target_id]), axis = 0)
               objectives =self.np.append(objectives, obj, axis = 0)
               case_id.append(case)

        return objectives, targets, case_id

    ### NORMALIZE ###
    def normalize_obj(self,objectives):
        minimum = []
        for i in range(len(self.DOF)):
            minimum.append(self.DOF[i]['min'])
        
        maximum = []
        for i in range(len(self.DOF)):
            maximum.append(self.DOF[i]['max'])
        
        for k in range(len(objectives)):
            for i in range(objectives[k].shape[0]):
                objectives[k][i] = (objectives[k][i] - minimum[i])/((maximum[i]-minimum[i]))
        return objectives

    def fit_data(self, solver = 'lbfgs', validation_fraction = 0.1, hidden_layer = (64,64), size_testdata = 20, activation = 'relu'):
        score = []
        self.mlp_classifier = []
        for area in range(len(self.area_names)):
            self.mlp_classifier.append('')
            objectives, targets, _ = self.extract_data(area)
            objectives = self.normalize_obj(objectives)
            if size_testdata != 0:
                len_obj = len(objectives)
                indices_test = list(range(len_obj-size_testdata, len_obj))
                objectives_test = objectives[indices_test,:]
                targets_test = targets[indices_test]
                ### Remove Test-Data from Trainingsdata ####
                objectives = self.np.delete(objectives, indices_test, axis=0)
                targets = self.np.delete(targets, indices_test)
                best_score = 0
                ### RUNNING 10 ERROR MINIMIZATION ###
                for i in range(10):
                    mlp = self.MLPClassifier(solver=solver, alpha=1e-5,hidden_layer_sizes=hidden_layer, max_iter = 20000, activation = activation, validation_fraction = validation_fraction)
                    mlp.fit(objectives,targets)
                    score_temp = mlp.score(objectives_test,targets_test)
                    if best_score < score_temp:
                        #print(f"New High-Score: {score_temp}")
                        #print(f"Run {i}.")
                        self.mlp_classifier[-1] = mlp
                        best_score = score_temp
                    if best_score == 1.0:
                        break
                test_score = 100*self.mlp_classifier[area].score(objectives_test,targets_test)
                #self.logger_cluster.info("")
                #self.logger_cluster.info(f'Accuracy: {100*self.mlp_classifier[area].score(objectives,targets):.2f}% for area {self.area_names[area]}.')
                #self.logger_cluster.info(f'Accuracy on Test-Data: {test_score:.2f}% for area {self.area_names[area]}.\n')
                print(f'Accuracy: {100*self.mlp_classifier[area].score(objectives,targets):.2f}% for area {self.area_names[area]}.')
                print(f'Accuracy on Test-Data: {test_score:.2f}% for area {self.area_names[area]}.')
                score.append(test_score/100)

            else:
                mlp = self.MLPClassifier(solver=solver, alpha=1e-5,hidden_layer_sizes=hidden_layer, max_iter = 20000, activation = activation, validation_fraction = validation_fraction)
                mlp.fit(objectives,targets)
                self.mlp_classifier[-1] = mlp
                #learning_center.PREDICTION_NET[-1] = mlp
                if area == 0:
                    #self.logger_cluster.debug("No Test-Data since Optimization Run.")
                    print("No Test-Data since Optimization Run.")
                #self.logger_cluster.info(f"Accuracy on Trainings-Data: {self.mlp_classifier[-1].score(objectives, targets)*100}% for area {self.area_names[area]}.")
                print(f"Accuracy on Trainings-Data: {self.mlp_classifier[-1].score(objectives, targets)*100}% for area {self.area_names[area]}.")
        learning_center.PREDICTION_NET = self.mlp_classifier
            

    def pred_cluster_id(self, objectives_pred_in):
        objectives_pred = self.copy.deepcopy(objectives_pred_in)
        objectives_pred = self.normalize_obj(objectives_pred)
        predicted_cluster_ids = self.np.empty((0,len(objectives_pred)))
        for area in range(len(self.area_names)):
            predicted_cluster_ids= self.np.append(predicted_cluster_ids, [learning_center.PREDICTION_NET[area].predict(objectives_pred)],  axis=0)
        predicted_cluster_ids = predicted_cluster_ids.transpose()
        predicted_cluster_ids = predicted_cluster_ids.astype(int)
        return predicted_cluster_ids


    def recommend_sim(self, objectives_in, train = True, dim = 'sing_val'):
        learning_center.EVAL_COUNTER += 1
        if learning_center.EVAL_COUNTER % 100 == 0:
            #self.logger_cluster.info(f"EVAL_COUNTER = {learning_center.EVAL_COUNTER}: RETRAIN.")
            print(f"EVAL_COUNTER = {learning_center.EVAL_COUNTER}: RETRAIN.")
            if train:
                print("RETRAINING AUTOENCODER.")
                #self.logger_cluster.info("RETRAINING AUTOENCODER.")
            self.retrain(self.anzahl_cluster, max_iter = 10, train = train)
        objectives_in = self.np.array(objectives_in)
        cluster_id_pred = self.pred_cluster_id(objectives_in)[0]
        #self.logger_cluster.info(f"EVAL CONTER = {learning_center.EVAL_COUNTER}")
        print(f"EVAL CONTER = {learning_center.EVAL_COUNTER}")
        bad_id = any([self.np.array_equal(cluster_id_pred, cluster_id_bad) for cluster_id_bad in learning_center.CLUSTER_IDS_BAD ])
        known_id = any([self.np.array_equal(cluster_id_pred, cluster_id) for cluster_id in learning_center.CLUSTER_IDS_VALUED[0]])

        eta_pred = []
        VCav_pred = []
        dH_pred = []
        P_pred = []
        neighbors = []
        gaussian = False
        fitness_pred_list = []

        if bad_id == True: 
            indice = [index for index, array in enumerate(learning_center.CLUSTER_IDS_VALUED[0]) if self.np.array_equal(array, cluster_id_pred)][0]
            fitness_pred = learning_center.CLUSTER_IDS_VALUED[3][indice]
            if len(learning_center.CLUSTER_IDS_VALUED[1][indice]) < 2:
                fitness_pred_list.append(fitness_pred[0])
            else:
                try:
                    fit_helper = self.pred_fit(cluster_id_pred, objectives_in[0], dim = dim)
                    fitness_pred_list.append(fit_helper)
                    #self.logger_cluster.info(f"Gaussian: {fit_helper}. Non-Gaussian: {fitness_pred[0]}.")
                    gaussian = True

                except Exception as e:
                    #self.logger_cluster.warning(e)
                    print(e)
                    fitness_pred_list.append(fitness_pred[0])

            eta_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[4], axis = 0)
            VCav_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[6], axis = 0)
            dH_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[5], axis = 0)
            P_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[7], axis = 0)
            neighbors = self.give_fitness(cluster_id_pred, output = 'off')[0]
            sim = False

        elif bad_id == False:
            sim = True
            fitness_pred = None
            eta_pred = None
            VCav_pred = None
            dH_pred = None
            P_pred = None

        if bad_id == False and known_id == True:
            indice = [index for index, array in enumerate(learning_center.CLUSTER_IDS_VALUED[0]) if self.np.array_equal(array, cluster_id_pred)][0]
            fitness_pred = learning_center.CLUSTER_IDS_VALUED[3][indice]
            if len(learning_center.CLUSTER_IDS_VALUED[1][indice]) < 2:
                fitness_pred_list.append(fitness_pred[0])
            else:
                fitness_pred_list.append(self.pred_fit(cluster_id_pred, objectives_in[0], dim = dim))
                #self.logger_cluster.info(f"Gaussian Estimation")
                gaussian = True

            eta_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[4], axis = 0)
            VCav_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[6], axis = 0)
            dH_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[5], axis = 0)
            P_pred = self.np.mean(self.give_fitness(cluster_id_pred, output = 'off')[7], axis = 0)
            neighbors = self.give_fitness(cluster_id_pred, output = 'off')[0]

        print(f"Recommend Sim: {sim}. Known Cluster_ID: {known_id}. Gaussian Estimation: {gaussian}.")
        if known_id:
            print(f"Predicted Fitness: {fitness_pred_list[0]}")
        return sim, known_id, fitness_pred_list, eta_pred, VCav_pred, dH_pred, P_pred, neighbors, gaussian


    def pred_fit(self, cluster_id, objectives, dim = 'sing_val'):
        obj = self.np.array(self.give_fitness(cluster_id, output = 'off')[1])
        fitness = self.np.array(self.give_fitness(cluster_id, output = 'off')[2])
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        ### PCA ###
        from sklearn.decomposition import PCA
        pca = PCA(n_components = min(len(obj),30), svd_solver='full')
        pca.fit(obj)
        #self.logger_cluster.info(f"Sing.Values: {pca.singular_values_}.")
        sing_val = pca.singular_values_
        n_relevant = len([s for s in sing_val if s > 5e-5])
        x_g = pca.transform(obj)
        x_input_transformed = pca.transform([objectives])[0]
        fit_return = []
        gpr_list = []
        kernel = 0.5 * RBF(0.6, length_scale_bounds='fixed')

        ### AVOID DIVIDE BY ZERO ###
        diffs = self.np.max(x_g, axis = 0) - self.np.min(x_g, axis = 0)
        indices = diffs >= 10e-6
        x_g = x_g[:,indices]
        x_input_transformed = x_input_transformed[indices]

        if x_g.shape[0] == 0:
            print("x_g.shape[0] == 0!")
            #self.logger_cluster.warning("x_g.shape[0] == 0!")
    
        if dim == 'auto':
            dim = min(len(obj),5)
        elif dim == 'sing_val':
            dim = min(n_relevant, len(obj))

        for i in range(3):
            y_g = fitness[:,i].reshape(-1,1)
            if max(y_g) - min(y_g) < 1e-8:
                fit_return.append(self.np.mean(y_g))
                #self.logger_cluster.warning(f"Took mean for y_g[{i}]")
            else:
                ## normalize ##
                y_g_tf = 2*(y_g - min(y_g))/((max(y_g)-min(y_g)))-1
                x_g_normed  = ((x_g  - self.np.min(x_g, axis = 0))/(self.np.max(x_g, axis =0) - self.np.min(x_g, axis =0)))[:,0:dim]
                x_input_transformed_normed =  ((x_input_transformed  - self.np.min(x_g, axis = 0))/(self.np.max(x_g, axis =0) - self.np.min(x_g, axis =0)))[0:dim]
                gpr = GaussianProcessRegressor(kernel = kernel,random_state=0, alpha = 10e-3).fit(x_g_normed, y_g_tf)       
                gpr_list.append(gpr)
                pred = gpr.predict(x_input_transformed_normed.reshape(1,-1), return_std=False)              
                ## denormalize ###
                if pred < -1:
                    self.logger_cluster.warning(f"Fitness_Pred_normed < -1: Setting to normed = -1.")
                    print(f"Fitness_Pred_normed < -1: Setting to normed = -1.")
                pred = max(-1,pred)
                pred =  (pred+1)/2 * (max(y_g)-min(y_g)) + min(y_g)
                fit_return.append(pred)
        fit_return = self.np.array(fit_return)
        fit_return = fit_return.reshape(1,-1)[0]
        #print(f'{dim}d')
        #self.logger_cluster.info(f"Gaussian Process with dim = {dim}.")
        print(f"Gaussian Process with dim = {dim}.")
        return fit_return
         

    def plot_dev(self, cluster_id_input, area = 1):
        self.plt.close('all')
        self.plt.figure(figsize=(10, 6))
        if isinstance(cluster_id_input, int):
            cluster_nummer = cluster_id_input
            state_liste = []
            for cluster_id in self.cluster_ids_valued[0]:
                if cluster_id[area - 1] == cluster_nummer:
                    if len(state_liste) == 0:
                        state_liste = self.give_fitness(cluster_id, output = 'off')[0]
                    else:
                        state_liste.append( self.give_fitness(cluster_id, output = 'off')[0][0])
            state_liste = [int(s.split("_")[1]) for s in state_liste] 
            index_list = [(state - 1) for state in state_liste]
            obj = self.case_data[1][index_list]
            obj_dev = self.np.std(obj, axis = 0)
            werte = [obj_dev[i]/(self.DOF[i]['max']-self.DOF[i]['min']) for i in range(0,30)]

        else:      
            cluster_nummer = [i for i in range(0,len(self.cluster_ids_valued[0])) if self.np.array_equal(cluster_id_input, self.cluster_ids_valued[0][i])][0]
            werte = [self.cluster_ids_valued[5][cluster_nummer][i]/(self.DOF[i]['max']-self.DOF[i]['min']) for i in range(0,30)]
            
        bezeichnung = [self.DOF[i]['label'] for i in range(0,30)]
        self.plt.barh(bezeichnung, werte,  height = 0.5)
        self.plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
        self.plt.xlabel('Standardabweichung genormt auf maximal erlaubte Differenz')
        self.plt.title(f"Standardabweichung der Objectives für {cluster_id_input}.")
        if max(werte) > 0.25:
            self.plt.xlim(0,max(werte)+0.1)
        else:
            self.plt.xlim(0, 0.25)

        self.plt.show()


    def plot_tensor_all(self, cluster_id, areas = [7,8,1,2,3,4,5,6], var = 'pressure'):
        stateNumbers = [int(s.split("_")[1]) for s in self.give_fitness(cluster_id, output = 'off')[0]]
        self.plt.close('all')
        fig = self.plt.figure(figsize=(14,8))
        if var == 'pressure':
            var_int = 0
        elif var == 'velo':
            var_int = [1,2,3]
        ax = fig.add_subplot(1,1,1, projection='3d')
        liste = ['coolwarm', 'viridis']
        for i in range(len(stateNumbers)):
            ind = self.np.where(self.T_denormalized[0] == (self.PREFIX + '_' + str(stateNumbers[i]) + self.op[:-1]))[0][0]
            if var == 'pressure':
                cat_list = [self.T_denormalized[i][ind][var_int,:,0,:] for i in areas]
            elif var == 'velo':
                cat_list = [self.T_denormalized[i][ind][var_int,:,0,:].norm(dim=0) for i in areas]           
            z_min = -150
            data = self.torch.cat(cat_list, dim=1)
            x = self.np.linspace(0, 1, data.shape[0])
            y_1d = self.np.linspace(0, 1, data.shape[1]) 
            x, y = self.np.meshgrid(x, y_1d, indexing='ij')
            ax.plot_surface(x, y, data, cmap='viridis', alpha = 0.4)

        if var == 'pressure':
            ax.set_zlabel('pressure')
        elif var == 'velo':
            ax.set_zlabel('[m/s]')
        if 8 in areas:
            y_Vorderkante = y_1d[(areas.index(8)+1)*14-1]
            x_line = self.np.linspace(0, 1, data.shape[0])
            y_line = self.np.ones_like(x_line)*y_Vorderkante
            z_line = self.np.full_like(x_line, z_min)
            ax.plot(x_line, y_line, z_line, color='black', linestyle='-', linewidth = 2.5)
            ax.text(1.5, y_Vorderkante ,z = z_min-10 ,s =  'leading edge', color='black', ha='center')

        if 4 in areas:
            y_Hinterkante = y_1d[(areas.index(4)+1)*14-1]
            x_line = self.np.linspace(0, 1, data.shape[0])
            y_line = self.np.ones_like(x_line)*y_Hinterkante
            z_line = self.np.full_like(x_line, z_min)
            ax.plot(x_line, y_line, z_line, color='black', linestyle='-', linewidth = 2.5)
            ax.text(1.5, y_Hinterkante ,z = z_min-10 ,s =  'trailing edge', color='black', ha='center')
        
        ax.set_xlabel('hub ---- shroud')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_zlim(-150, 100)
        ax.set_box_aspect([1, 3, 1])
        ax.view_init(elev=20, azim=-30)
        if var == 'pressure':
            ax.set_zlabel('pressure')
        elif var == 'velo':
            ax.set_zlabel('[m/s]')
        self.plt.title(f"{var}-fields of Cluster_ID: {cluster_id}")
        self.plt.yticks([])
        self.plt.tight_layout
        self.plt.show()
         
               
    def retrain(self, cluster_anzahl, max_iter = 50, train = True):
        #self.logger_cluster.info("Start Training...")
        print("*** Start Retrain ***")
        self.load_tensors(normalize = True)
        self.load_model(model_type = 'clustering', pretrained = True)
        if train == True:
            self.train_model_c_all(max_iter = max_iter)
        self.get_latent_data()
        self.set_cluster_anzahl(cluster_anzahl)
        self.get_cluster_ids(algo = self.cl_alg, mode = 'all')
        self.fit_data(solver='lbfgs', validation_fraction=0.1, hidden_layer = (64,54), size_testdata = 15, activation = 'relu' )
        print("*** Design Assisant Retraining Finished ***")


    def get_z(self, stateNumber):
        case = self.PREFIX + '_' + str(stateNumber)
        indice = self.np.where(self.z[0] == case)[0]
        z = self.z[1][indice]
        return z



    def plot_tensor_one(self, cluster_id, areas = [7,8,1,2,3,4,5,6], var = 'pressure'):
            from mpl_toolkits.mplot3d import Axes3D
            stateNumbers = [int(s.split("_")[1]) for s in self.give_fitness(cluster_id, output = 'off')[0]]
            self.plt.close('all')
            fig = self.plt.figure(figsize = [10, 8])
            number_plots_y = 2
            number_plots_x = int(self.np.ceil(len(stateNumbers)/number_plots_y))
            if var == 'pressure':
                var_int = 0
            elif var == 'velo':
                var_int = [1,2,3]
            ax = fig.add_subplot(1,1,1, projection='3d')
            liste = ['coolwarm', 'viridis']
            for i in range(len(stateNumbers)):
                ind = self.np.where(self.T_denormalized[0] == (self.PREFIX + '_' + str(stateNumbers[i]) + self.op[:-1]))[0][0]
                if var == 'pressure':
                    cat_list = [self.T_denormalized[i][ind][var_int,:,0,:] for i in areas]
                elif var == 'velo':
                    cat_list = [self.T_denormalized[i][ind][var_int,:,0,:].norm(dim=0) for i in areas]           
            
                z_min = -150
                data = self.torch.cat(cat_list, dim=1)
                x = self.np.linspace(0, 1, data.shape[0])
                y_1d = self.np.linspace(0, 1, data.shape[1]) 
                x, y = self.np.meshgrid(x, y_1d, indexing='ij')
                ax.plot_surface(x, y, data, cmap='viridis', alpha = 0.4)

            #ax.set_ylabel('s ->') #: Entgegen Uhrzeigersinn
            if var == 'pressure':
                ax.set_zlabel('pressure' ,fontsize = 22)
            elif var == 'velo':
                ax.set_zlabel('[m/s]')
            if 8 in areas:
                y_Vorderkante = y_1d[(areas.index(8)+1)*14-1]
                x_line = self.np.linspace(0, 1, data.shape[0])
                y_line = self.np.ones_like(x_line)*y_Vorderkante
                z_line = self.np.full_like(x_line, z_min)
                ax.plot(x_line, y_line, z_line, color='black', linestyle='-', linewidth = 2.5)
                ax.text(1.5, y_Vorderkante ,z = z_min-10 ,s =  'leading edge', fontsize = 22, color='black', ha='center')

            if 4 in areas:
                y_Hinterkante = y_1d[(areas.index(4)+1)*14-1]
                x_line = self.np.linspace(0, 1, data.shape[0])
                y_line = self.np.ones_like(x_line)*y_Hinterkante
                z_line = self.np.full_like(x_line, z_min)
                ax.plot(x_line, y_line, z_line, color='black', linestyle='-', linewidth = 2.5)
                ax.text(1.5, y_Hinterkante ,z = z_min-10 ,s =  'trailing edge', fontsize = 22, color='black', ha='center')
            
            ax.set_xlabel('hub ---- shroud',fontsize = 22)
            #ax.set_ylabel('s ->') #: Entgegen Uhrzeigersinn
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.set_zlim(-150, 100)
            ax.set_box_aspect([1, 3, 1])
            ax.view_init(elev=20, azim=-30)
            #self.plt.title(f"Cluster_ID: {cluster_id}")
            self.plt.yticks([])
            self.plt.tight_layout
            self.plt.show()
























            
            
        
        









