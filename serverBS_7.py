# serverBS_7 features:
#       - upgrade from serverBS_4_2 (only update weights for topmost layers)
#       - change to enable updating weights for only the specified layers
#       - added parameter p for sigma power

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import copy
import torch
import random
class FedBS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        # self.load_model()
        self.Budget = []
        self.model_sigma = None
        self.local_bias = []
        self.old_model = None
        for _ in range(self.num_clients):
            local_bias = copy.deepcopy(self.global_model)
            for param in local_bias.parameters():
                param.data.zero_()
            self.local_bias.append(local_bias)
        self.sigma_lr =0.01
        self.p = 2
        self.clamp_val = 10
        self.selected_layers = [4, 5, 6, 7]
        print(f"\nParameter settings:")
        print(f"\n  - sigma_lr          : {self.sigma_lr}")
        print(f"\n  - p                 : {self.p}")
        print(f"\n  - clamp_val         : {self.clamp_val}")
        print(f"\n  - selected layers   : {self.selected_layers}")
 
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models_BS()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.old_model = copy.deepcopy(self.global_model)
            self.aggregate_parameters()
            self.generate_sigma()
            self.update_bias()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def generate_sigma(self):
        assert (len(self.uploaded_models) > 0)
        # Calculate sd of uploaded parameters
        if self.model_sigma == None:
            return
        for param in [list(self.model_sigma.parameters())[i] for i in self.selected_layers]:
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for sigma, client_param, global_param in [list(zip(self.model_sigma.parameters(), client_model.parameters(), self.global_model.parameters()))[i] for i in self.selected_layers]:
                # print(sigma.data)
                sigma.data += pow((client_param.data - global_param.data), 2)*w
        for sigma in [list(self.model_sigma.parameters())[i] for i in self.selected_layers]:
            sigma.data = torch.sqrt(sigma.data)

    # def generate_sigma(self):
    #     assert(len(self.uploaded_models) > 0)
    #     params = [list(model.parameters()) for model in self.uploaded_models]
    #     params = list(map(list, zip(*params)))
    #     params = [params[i] for i in self.selected_layers]
    #     self.model_sigma = [torch.std(torch.stack(param), dim=0) for param in params]
        
    def update_bias(self):
        for uploaded_model, id in zip(self.uploaded_models, self.uploaded_ids):
            for bias, local_param, global_param, sigma in [list(zip(self.local_bias[id].parameters(), uploaded_model.parameters(), self.global_model.parameters(), self.model_sigma.parameters()))[i] for i in self.selected_layers]:
                bias_update = (local_param.data-global_param.data) / pow(sigma.data, self.p-1)
                bias_update[torch.isnan(bias_update)] = 0.0
                torch.clamp(bias_update, min=-self.clamp_val, max=self.clamp_val)  
                bias.data = (1-self.sigma_lr*pow(sigma.data, 2-self.p)) * bias.data + self.sigma_lr*bias_update

    def send_models_BS(self):
        assert (len(self.clients) > 0)
        if self.model_sigma == None:
            self.model_sigma = copy.deepcopy(self.global_model)
            self.send_models()
        else:
            for client in self.clients:
                start_time = time.time()
                local_model = copy.deepcopy(self.global_model)
                for param, bias, sigma in [list(zip(local_model.parameters(), self.local_bias[client.id].parameters(), self.model_sigma.parameters()))[i] for i in self.selected_layers]:
                    param.data += bias.data*sigma.data
                client.set_parameters(local_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

