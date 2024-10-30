import json
class MyDataLoader:
    def __init__(self, module_name, path_dict, base_path=""):
        self.arcprize = {}
        if module_name == "arckit":
            pass
        
        if module_name == "arcprize":
            if path_dict == None:
                raise Exception('No data')
                
            for path_str in path_dict:
                file_path = base_path + path_dict[path_str]
                with open(file_path, 'r') as file:
                    self.arcprize[path_str] = json.load(file)
                    print("{} is loaded: {}".format(path_str, file_path))
                    
    def cur_data_mode(self, data_mode):
        if data_mode == "train":
            self.cur_problem     = self.arcprize["train_problem"]
            self.cur_target_goal = self.arcprize["train_target_goal"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
            
        if data_mode == "evaluation":
            self.cur_problem     = self.arcprize["evaluation_problem"]
            self.cur_target_goal = self.arcprize["evaluation_target_goal"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
            
        if data_mode == "test":
            self.cur_problem     = self.arcprize["test_problem"]
            self.cur_target_goal = self.arcprize["predicted_target_goal"]
            self.cur_idx = self.cur_problem.keys()
            print("{} data is set".format(data_mode))
    def get_data(self):
        return {
            "train:" : self.cur_problem,
            "solution": self.cur_target_gol
        }