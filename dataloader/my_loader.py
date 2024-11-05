import json
import os
import glob
from dataloader.my_data_caller import split_dict

class MyDataLoader:
    def __init__(self, module_name, path_ref, base_path="./data/kaggle/"):
        self.base_dirctory = "./data/dataset"
        self.base_name = '_data_file.json'
        self.settings_path = "./settings"
        self.arcprize = {}
        self.userdataset = {}
        self.datasets = {}
        self.module_name = module_name
        # we are not using arckit yet
        if module_name == "arckit":
            pass
        if path_ref == None:
            raise Exception('No data')
            
        if module_name == "arcprize":
            for path_str in path_ref:
                file_path = base_path + path_ref[path_str]
                with open(file_path, 'r') as file:
                    self.arcprize[path_str] = json.load(file)
                    print("{} is loaded: {}".format(path_str, file_path))
           

        if module_name == "userdata":
        
            for userdataset in path_ref:
                for userdata in path_ref[userdataset]:
                    for taskId in path_ref[userdataset][userdata]:
                        newId = taskId
                        if taskId in self.datasets:
                            print("Error: Id is duplicated", userdataset, userdata, taskId)
                            print("Status: Alread existed on, we Chanage id",  path_ref[userdataset][userdata][taskId])
                            newId += userdata
                        
                        with open(path_ref[userdataset][userdata][taskId], 'r') as file:
                            self.datasets[newId] = json.load(file)
                            
                train, evaluation, test = split_dict(self.datasets)
                self.userdataset["train"] = train
                self.userdataset["evaluation"] = evaluation
                self.userdataset["test"] = test
                print("{} is loaded! {}".format(userdataset, "data/dataset/"+userdataset))
             


    def cur_data_mode(self, data_mode):
        if self.module_name == 'userdata':
            self.arc_user_data_mode(data_mode)
        if self.module_name == 'arcprize':                 
            self.arc_prize_data_mode(data_mode)

    def arc_prize_data_mode(self, data_mode):
        if data_mode == "train":
            self.cur_problem = self.arcprize["train_problem"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
            
        if data_mode == "evaluation":
            self.cur_problem     = self.arcprize["evaluation_problem"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
            
        if data_mode == "test":
            self.cur_problem     = self.arcprize["test_problem"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
    
    def arc_user_data_mode(self, data_mode):
        if data_mode == "train":
            self.cur_problem = self.userdataset["train"]
            self.cur_idx = self.cur_problem.keys()
            print("{} user data is set".format(data_mode))
            
        if data_mode == "evaluation":
            self.cur_problem     = self.userdataset["evaluation"]
            self.cur_idx = self.cur_problem.keys()
            print("{} user data is set".format(data_mode))
            
        if data_mode == "test":
            self.cur_problem     = self.userdataset["test"]
            self.cur_idx = self.cur_problem.keys()
            print("{} user data is set".format(data_mode))
    
    def arc_data_collection_mode(self, data_collection):
        self.cur_problem = self.arc_collection_setting[data_collection]
        self.cur_idx = self.cur_problem.keys()
        print("{} data is set {} ".format(data_collection, self.arc_collection_setting[data_collection]))

    # 나중에 이건 아예 따로 분리할 것 데이터 로더랑 하기에는 적절하지 않은 것 같다.
    def arc_data_collection_figure(self,setting_directory='settings/datapathType2', base_directory='data/datasets'):
        # Dictionary to store JSON file paths and the dataset names
        json_info = {}

        # Use glob to find all JSON files under 'data/datasets'
        json_files = glob.glob(os.path.join(base_directory, "*", "**", "*.json"), recursive=True)
        # json_files = glob.glob(os.path.join(base_directory, "**", "*.json"), recursive=True)
        print(json_files)

        for json_file in json_files:
            # Extract the relative path starting from 'data/datasets'
            relative_path = os.path.relpath(json_file, base_directory)
        
            # Split the relative path to get the dataset name
            path_parts = relative_path.split(os.sep)
            dataset_name = path_parts[0]  # This should be the dataset name under 'data/datasets'

            # Store the file path and its dataset name
            json_info[ dataset_name ] = {}
        # Display results

        # Process each JSON file found
        for json_file in json_files:
            # Extract the relative path starting from 'data/datasets'
            relative_path = os.path.relpath(json_file, base_directory)
            immediate_folder = os.path.basename(os.path.dirname(json_file))

            # Split the relative path to get the dataset name
            path_parts = relative_path.split(os.sep)
            dataset_name = path_parts[0]  # This should be the dataset name under 'data/datasets'

            # Store the file path and its dataset name
            json_info[ dataset_name ][immediate_folder] = json_file
            
        file_list = {}
        # Display results
        for key in json_info:
            with open(setting_directory+'/'+key+self.base_name, 'w') as datafile:
                json.dump(json_info[key], datafile, indent=4)
                json_files = glob.glob(os.path.join(base_directory, "*.json"), recursive=True)
                # json_files = glob.glob(os.path.join(base_directory, "**", "*.json"), recursive=True)
                for json_file in json_files:
                    file_list[json_file[len(self.settings_path)+1:(-len(self.base_name))]] = json_info[key]
        with open(setting_directory+'/arc_collections.json', 'w') as datafile:
            json.dump(file_list, datafile, indent=4)


        # with open('./settings/arc_data_collection_file_name.json', 'r') as datafile:
        #     arc_collection_setting = json.load(datafile)