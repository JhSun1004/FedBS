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
from flcore.clients.clientBS import clientBS
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
        self.set_clients(clientBS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.model_sigma = None
        self.local_bias = []
        self.uploaded_gradients = []
        for _ in range(self.num_clients):
            local_bias = copy.deepcopy(self.global_model)
            for param in local_bias.parameters():
                param.data.zero_()
            self.local_bias.append(local_bias)
        self.sigma_lr =0.01

    def train(self):
        sigma_lr = self.sigma_lr
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

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.generate_sigma()
            sigma_lr = self.sigma_lr    #/(i//20+1)
            self.update_bias(sigma_lr)
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
        # 计算上传参数的标准差
        if self.model_sigma == None:
            return
        for param in self.model_sigma.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for sigma, client_param, global_param in zip(self.model_sigma.parameters(), client_model.parameters(), self.global_model.parameters()):
                sigma.data += pow((client_param.data - global_param.data), 2)*w
        for sigma in self.model_sigma.parameters():
            sigma.data = torch.sqrt(sigma.data)

    def update_bias(self, sigma_lr):
        for uploaded_model, id in zip(self.uploaded_models, self.uploaded_ids):
            for bias, local_param, global_param, sigma in zip(self.local_bias[id].parameters(), uploaded_model.parameters(), self.global_model.parameters(), self.model_sigma.parameters()):
                bias.data = (1-sigma_lr)*bias.data \
                            + sigma_lr*self.div(local_param.data-global_param.data, sigma.data)

    def send_models_BS(self):
        assert (len(self.clients) > 0)
        if self.model_sigma == None:
            self.model_sigma = copy.deepcopy(self.global_model)
            self.send_models()
        else:
            for client in self.clients:
                start_time = time.time()
                local_model = copy.deepcopy(self.global_model)
                for param, bias, sigma in zip(local_model.parameters(), self.local_bias[client.id].parameters(), self.model_sigma.parameters()):
                    param.data += bias.data*sigma.data
                client.set_parameters(local_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_BS(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_gradients = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
                self.uploaded_gradients.append(client.gradient)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
    def div(self, substract, sigma_data):
        substract /= sigma_data
        substract[torch.isnan(substract)] = 0.0
        substract[torch.gt(substract, 10)] = 0.0
        return substract
