

# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientavg import clientAVG
from flcore.clients.clientala import clientALA
from flcore.servers.serverbase import Server
from threading import Thread
import copy
import matplotlib.pyplot as plt
import torch
import numpy as np

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
        self.mean_sigma = []
        self.bias_layer = [1, 6]
        # self.bias_layer = range(8)
        local_bias = []
        for i, param in enumerate(self.global_model.parameters()):
            if i in self.bias_layer:
                local_bias.append(copy.deepcopy(param))
            
        for _ in range(self.num_clients):
            self.local_bias.append(copy.deepcopy(local_bias))
        self.sigma_lr = 1

    def train(self):
        for i in range(self.global_rounds+1):
            # if (i + 1) % 20 == 0:
            #     self.sigma_lr *= 0.7
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models_BS()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

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
        # self.mean_sigma = np.array(self.mean_sigma).T
        # for i in range(8):
        #     plt.plot(np.array(range(self.global_rounds+1)),self.mean_sigma[i], label=f'layer {i}')
        # plt.xlabel('rounds')
        # plt.ylabel('mean_sigma')
        # plt.legend()
        # plt.title('Mean sigma variation with rounds')
        # plt.savefig(f'sigma{self.sigma_lr}.png')

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    # def generate_sigma(self):
    #     assert(len(self.uploaded_models) > 0)
    #     # Calculate sd of uploaded parameters
    #     if self.model_sigma == None:
    #         return
    #     for param in self.model_sigma:
    #         param.data.zero_()
    #     params_g = list(self.global_model.parameters())
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         params_l = list(client_model.parameters())
    #         for i, sigma in enumerate(self.model_sigma):
    #             sigma.data += pow((params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data), 2) * w
    #     for sigma in self.model_sigma:
    #         sigma.data = torch.sqrt(sigma.data)

        
    def generate_sigma(self):
        assert(len(self.uploaded_models) > 0)
        # Calculate sd of uploaded parameters
        if self.model_sigma == None:
            return
        for param in self.model_sigma:
            param.data.zero_()
        params_g = list(self.global_model.parameters())
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            params_l = list(client_model.parameters())
            for i, sigma in enumerate(self.model_sigma):
                sigma.data += abs(params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) * w
        # sigmas = []
        # for sigma in self.model_sigma:
        #     sigmas.append(torch.mean(sigma).item())
        # print(f"Sigma: {sigmas}")
        # self.mean_sigma.append(sigmas)

    # def generate_sigma(self):
    #     assert (len(self.uploaded_models) > 0)
    #     # Calculate sd of uploaded parameters
    #     if self.model_sigma == None:
    #         return
    #     for param in self.model_sigma.parameters():
    #         param.data.zero_()
    #     for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
    #         for sigma, client_param, global_param in zip(self.model_sigma.parameters(), client_model.parameters(), self.global_model.parameters()):
    #             sigma.data += pow((client_param.data - global_param.data), 2) * w
    #     for sigma in self.model_sigma.parameters():
    #         sigma.data = torch.sqrt(sigma.data)
    #     sigmas = []
    #     for sigma in self.model_sigma.parameters():
    #         sigmas.append(torch.mean(sigma).item())
    #     print(f"Sigma: {sigmas}")
    #     self.mean_sigma.append(sigmas)

    def update_bias(self):
        params_g = list(self.global_model.parameters())
        for uploaded_model, id in zip(self.uploaded_models, self.uploaded_ids):
            params_l = list(uploaded_model.parameters())
            # for bias, local_param, global_param, sigma in zip(self.local_bias[id].parameters(), uploaded_model.parameters(), self.old_model.parameters(), self.model_sigma.parameters()):
            #     bias.data = self.sigma_lr * (local_param.data - global_param.data) + (1 - self.sigma_lr * sigma.data) * bias.data
            for i, (bias, sigma) in enumerate(zip(self.local_bias[id], self.model_sigma)):
                bias.data = self.sigma_lr * (params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) + (1 - self.sigma_lr * sigma.data) * bias.data


    # def update_bias(self):
    #     for uploaded_model, id in zip(self.uploaded_models, self.uploaded_ids):
    #         for bias, local_param, global_param, sigma in zip(self.local_bias[id].parameters(), uploaded_model.parameters(), self.global_model.parameters(), self.model_sigma.parameters()):
    #             bias.data = self.sigma_lr * (local_param.data - global_param.data) + (1 - self.sigma_lr * sigma.data) * bias.data
    #             # bias.data = local_param.data - global_param.data

    def send_models_BS(self):
        assert (len(self.clients) > 0)
        if self.model_sigma == None:
            # self.model_sigma = copy.deepcopy(self.global_model)
            self.model_sigma = []
            for i, param in enumerate(self.global_model.parameters()):
                if i in self.bias_layer:
                    self.model_sigma.append(copy.deepcopy(param))
            self.send_models()
        else:
                
            for client in self.clients:
                start_time = time.time()
                local_model = copy.deepcopy(self.global_model)
                params = list(local_model.parameters())
                # for i, param, bias, sigma in zip(local_model.parameters(), self.local_bias[client.id].parameters(), self.model_sigma.parameters()):
                #     param.data += bias.data * sigma
                for i, (bias, sigma) in enumerate(zip(self.local_bias[client.id], self.model_sigma)):
                    params[self.bias_layer[i]].data += bias.data * sigma.data
                client.set_parameters(local_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
