import torch
import copy
import torch.nn as nn
import copy
class BS:
    def __init__(self, global_model: torch.nn.Module, sigma_lr: float = 1, num_clients: int = 20, sigma_type: int = 2):
        """
        Initialize the Bias Correction module.

        Args:
            global_model: The global model.
            sigma_lr: The learning rate for the sigma. Default: 1

        Returns:
            None.
        """
        self.sigma_lr = sigma_lr
        self.sigma_type = sigma_type
        self.global_model = copy.deepcopy(global_model)
        self.model_sigma = None
        self.bias_layer = [1, 6]
        self.local_bias = []
        local_bias = []
        for i, param in enumerate(global_model.parameters()):
            if i in self.bias_layer:
                local_bias.append(copy.deepcopy(param))
        for _ in range(num_clients):
            self.local_bias.append(copy.deepcopy(local_bias))
    
    def generate_sigma_1(self, global_model: torch.nn.Module, uploaded_models: list, uploaded_weights: list):
        """
        Generate the sigma for the bias correction.

        Args:
            None.

        Returns:
            None.
        """
        assert(len(self.uploaded_models) > 0)
        if self.model_sigma == None:
            return
        for sigma in self.model_sigma:
            sigma.data.zero_()
        params_g = list(self.global_model.parameters())
        for w, client_model in zip(uploaded_weights, uploaded_models):
            params_l = list(client_model.parameters())
            for i, sigma in enumerate(self.model_sigma):
                sigma.data += abs(params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) * w

    def generate_sigma_2(self, global_model: torch.nn.Module, uploaded_models: list, uploaded_weights: list):
        """
        Generate the sigma for the bias correction.

        Args:
            None.

        Returns:
            None.
        """
        assert(len(uploaded_models) > 0)
        # Calculate sd of uploaded parameters
        if self.model_sigma == None:
            return
        for sigma in self.model_sigma:
            sigma.data.zero_()
        params_g = list(global_model.parameters())
        for w, client_model in zip(uploaded_weights, uploaded_models):
            params_l = list(client_model.parameters())
            for i, sigma in enumerate(self.model_sigma):
                sigma.data += pow((params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data), 2) * w
        for sigma in self.model_sigma:
            sigma.data = torch.sqrt(sigma.data)
        
    def update_bias(self, uploaded_models: list, uploaded_ids: list):
        params_g = list(self.global_model.parameters())
        for uploaded_model, id in zip(uploaded_models, uploaded_ids):
            params_l = list(uploaded_model.parameters())
            for i, (bias, sigma) in enumerate(zip(self.local_bias[id], self.model_sigma)):
                bias.data = self.sigma_lr * (params_l[self.bias_layer[i]].data - params_g[self.bias_layer[i]].data) + (1 - self.sigma_lr * sigma.data) * bias.data

    def update(self, global_model: torch.nn.Module, uploaded_models: list, uploaded_weights: list, uploaded_ids: list):
        if self.model_sigma == None:
            # self.model_sigma = copy.deepcopy(self.global_model)
            self.model_sigma = []
            for i, param in enumerate(self.global_model.parameters()):
                if i in self.bias_layer:
                    self.model_sigma.append(copy.deepcopy(param))
        if self.sigma_type == 1:
            self.generate_sigma_1(global_model, uploaded_models, uploaded_weights)
        else:
            self.generate_sigma_2(global_model, uploaded_models, uploaded_weights)                       
        self.update_bias(uploaded_models, uploaded_ids)
        self.global_model = copy.deepcopy(global_model)

    def distribute_model(self, client):
        """
        Distribute the model to the client.

        Args:
            client: The client to distribute the model to.
            local_model: The local model to distribute.

        Returns:
            None.
        """
        if self.model_sigma == None:
            client.set_parameters(self.global_model)
        else:  
            local_model = copy.deepcopy(self.global_model)
            params = list(local_model.parameters())
            for i, (bias, sigma) in enumerate(zip(self.local_bias[client.id], self.model_sigma)):
                params[self.bias_layer[i]].data += bias.data * sigma.data
            client.set_parameters(local_model)


