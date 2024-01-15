import torch
import pandas as pd
import os

class datastorage():
    def __init__(self, seed_list, init_list, compression_list, data_list, method_list):
        
        self.methodlist = method_list
        self.seedlist = seed_list  #[1,15,85]
        self.init = init_list   # ["kcenter", "herding"]
        self.data = data_list    #["cora", "citeseer"]
        self.compression = compression_list # [0.25,0.5]

        # create tables 3seeds - 1,15,85 
        self.results = self.create_table()
        output_folder = 'init_results_files'
        os.makedirs(output_folder, exist_ok=True)
        self.path = "init_results_files/results-in-red.csv"
        
        self.i =0

    def load_table(self):
        if os.path.isfile(self.path):
            
            df= pd.read_csv(self.path,header=None)

            df.columns = self.col_idx

            df.set_index(self.row_idx, inplace=True)

            self.results =df
        
    def create_table(self):
        idx1 = []
        idx2= []
        idx3=[]
        for i in self.data:
            for j in self.compression[i]:
                for k in self.init:
                    idx1.append(i)
                    idx2.append(j)
                    idx3.append(k)
        
        self.col_idx = pd.MultiIndex.from_product([self.methodlist,self.seedlist],names = ["Methods","Seeds"])
        self.row_idx = pd.MultiIndex.from_arrays([idx1,idx2, idx3],names = ["Dataset","Compression ratio", "Initialization"])
        df = pd.DataFrame(columns=self.col_idx, index=self.row_idx)
        df = df.fillna(0)
        return df

    def update(self, results, details):
        
        dataname, compress,init,method, seed = details

        self.results.loc[(dataname, compress,init),(method, seed)] = results
            
        self.i+=1
        if self.i%5==0:
            self.display_save()


    def display_save(self): 
        # save all seeds files        
        self.results.to_csv(self.path,header=False, index=False)