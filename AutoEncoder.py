# Marcus Vinicius Sousa Leite de Carvalho
# marcus.decarvalho@ntu.edu.sg
# ivsucram@gmail.com
#
# NANYANG TECHNOLOGICAL UNIVERSITY - NTUITIVE PTE LTD Dual License Agreement
# Non-Commercial Use Only
# This NTUITIVE License Agreement, including all exhibits ("NTUITIVE-LA") is a legal agreement between you and NTUITIVE (or “we”) located at 71 Nanyang Drive, NTU Innovation Centre, #01-109, Singapore 637722, a wholly owned subsidiary of Nanyang Technological University (“NTU”) for the software or data identified above, which may include source code, and any associated materials, text or speech files, associated media and "online" or electronic documentation and any updates we provide in our discretion (together, the "Software").
#
# By installing, copying, or otherwise using this Software, found at https://github.com/Ivsucram/ATL_Matlab, you agree to be bound by the terms of this NTUITIVE-LA.  If you do not agree, do not install copy or use the Software. The Software is protected by copyright and other intellectual property laws and is licensed, not sold.   If you wish to obtain a commercial royalty bearing license to this software please contact us at marcus.decarvalho@ntu.edu.sg.
#
# SCOPE OF RIGHTS:
# You may use, copy, reproduce, and distribute this Software for any non-commercial purpose, subject to the restrictions in this NTUITIVE-LA. Some purposes which can be non-commercial are teaching, academic research, public demonstrations and personal experimentation. You may also distribute this Software with books or other teaching materials, or publish the Software on websites, that are intended to teach the use of the Software for academic or other non-commercial purposes.
# You may not use or distribute this Software or any derivative works in any form for commercial purposes. Examples of commercial purposes would be running business operations, licensing, leasing, or selling the Software, distributing the Software for use with commercial products, using the Software in the creation or use of commercial products or any other activity which purpose is to procure a commercial gain to you or others.
# If the Software includes source code or data, you may create derivative works of such portions of the Software and distribute the modified Software for non-commercial purposes, as provided herein.
# If you distribute the Software or any derivative works of the Software, you will distribute them under the same terms and conditions as in this license, and you will not grant other rights to the Software or derivative works that are different from those provided by this NTUITIVE-LA.
# If you have created derivative works of the Software, and distribute such derivative works, you will cause the modified files to carry prominent notices so that recipients know that they are not receiving the original Software. Such notices must state: (i) that you have changed the Software; and (ii) the date of any changes.
#
# You may not distribute this Software or any derivative works.
# In return, we simply require that you agree:
# 1.    That you will not remove any copyright or other notices from the Software.
# 2.    That if any of the Software is in binary format, you will not attempt to modify such portions of the Software, or to reverse engineer or decompile them, except and only to the extent authorized by applicable law.
# 3.    That NTUITIVE is granted back, without any restrictions or limitations, a non-exclusive, perpetual, irrevocable, royalty-free, assignable and sub-licensable license, to reproduce, publicly perform or display, install, use, modify, post, distribute, make and have made, sell and transfer your modifications to and/or derivative works of the Software source code or data, for any purpose.
# 4.    That any feedback about the Software provided by you to us is voluntarily given, and NTUITIVE shall be free to use the feedback as it sees fit without obligation or restriction of any kind, even if the feedback is designated by you as confidential.
# 5.    THAT THE SOFTWARE COMES "AS IS", WITH NO WARRANTIES. THIS MEANS NO EXPRESS, IMPLIED OR STATUTORY WARRANTY, INCLUDING WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, ANY WARRANTY AGAINST INTERFERENCE WITH YOUR ENJOYMENT OF THE SOFTWARE OR ANY WARRANTY OF TITLE OR NON-INFRINGEMENT. THERE IS NO WARRANTY THAT THIS SOFTWARE WILL FULFILL ANY OF YOUR PARTICULAR PURPOSES OR NEEDS. ALSO, YOU MUST PASS THIS DISCLAIMER ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
# 6.    THAT NEITHER NTUITIVE NOR NTU NOR ANY CONTRIBUTOR TO THE SOFTWARE WILL BE LIABLE FOR ANY DAMAGES RELATED TO THE SOFTWARE OR THIS NTUITIVE-LA, INCLUDING DIRECT, INDIRECT, SPECIAL, CONSEQUENTIAL OR INCIDENTAL DAMAGES, TO THE MAXIMUM EXTENT THE LAW PERMITS, NO MATTER WHAT LEGAL THEORY IT IS BASED ON. ALSO, YOU MUST PASS THIS LIMITATION OF LIABILITY ON WHENEVER YOU DISTRIBUTE THE SOFTWARE OR DERIVATIVE WORKS.
# 7.    That we have no duty of reasonable care or lack of negligence, and we are not obligated to (and will not) provide technical support for the Software.
# 8.    That if you breach this NTUITIVE-LA or if you sue anyone over patents that you think may apply to or read on the Software or anyone's use of the Software, this NTUITIVE-LA (and your license and rights obtained herein) terminate automatically.  Upon any such termination, you shall destroy all of your copies of the Software immediately.  Sections 3, 4, 5, 6, 7, 8, 11 and 12 of this NTUITIVE-LA shall survive any termination of this NTUITIVE-LA.
# 9.    That the patent rights, if any, granted to you in this NTUITIVE-LA only apply to the Software, not to any derivative works you make.
# 10.   That the Software may be subject to U.S. export jurisdiction at the time it is licensed to you, and it may be subject to additional export or import laws in other places.  You agree to comply with all such laws and regulations that may apply to the Software after delivery of the software to you.
# 11.   That all rights not expressly granted to you in this NTUITIVE-LA are reserved.
# 12.   That this NTUITIVE-LA shall be construed and controlled by the laws of the Republic of Singapore without regard to conflicts of law.  If any provision of this NTUITIVE-LA shall be deemed unenforceable or contrary to law, the rest of this NTUITIVE-LA shall remain in full effect and interpreted in an enforceable manner that most nearly captures the intent of the original language.
#
# Copyright (c) NTUITIVE. All rights reserved.

from NeuralNetwork import NeuralNetwork
from MySingletons import MyDevice

import numpy as np
import torch


class AutoEncoder(NeuralNetwork):
    _greedy_layer_bias = None
    _greedy_layer_output_bias = None

    @property
    def latent_space(self):
        return self.layer_value[self.latent_space_position]

    @property
    def latent_space_size(self):
        return self.layers[self.latent_space_position]

    @property
    def latent_space_position(self):
        return int((len(self.layers) - 1) / 2)

    def __init__(self, layers=[]):
        NeuralNetwork.__init__(self, layers)
        for i in range(self.number_hidden_layers):
            self.activation_function[i] = self.ACTIVATION_FUNCTION_SIGMOID
        self.output_activation_function = self.ACTIVATION_FUNCTION_SIGMOID
        self.loss_function = self.LOSS_FUNCTION_MSE

    def train(self, x: torch.tensor, is_tied_weight: bool = False, noise_ratio: float = 0.0, weight_number: int = None, y: torch.tensor = None):
        if is_tied_weight:
            for i in range(int(self.number_hidden_layers/2)):
                if i == 0:
                    self.output_weight = self.weight[i].T
                else:
                    self.weight[-i] = self.weight[i].T

        if y is None:
            y = x
        NeuralNetwork.train(self, x=self.masking_noise(x=x, noise_ratio=noise_ratio), y=y, weight_no=weight_number)

    def test(self, x: torch.tensor, is_beta_updatable: bool = False, y: torch.tensor = None):
        if y is None:
            y = x
        return NeuralNetwork.test(self, x=x, y=y, is_beta_updatable=is_beta_updatable)

    def grow_node(self, layer_number):
        NeuralNetwork.grow_node(self, layer_number)
        self.grow_greedy_layer_bias(layer_number)

    def prune_node(self, layer_number, node_number):
        NeuralNetwork.prune_node(self, layer_number, node_number)
        self.prune_greedy_layer_bias(layer_number, node_number)

    def grow_greedy_layer_bias(self, layer_number):
        b = layer_number
        if b is self.number_hidden_layers:
            [n_out, n_in] = self._greedy_layer_output_bias.shape
            self._greedy_layer_output_bias = torch.cat((self._greedy_layer_output_bias, self.xavier_weight_initialization(1, 1)), axis=1)
        else:
            [n_out, n_in] = self._greedy_layer_bias[b].shape
            n_in = n_in + 1
            self._greedy_layer_bias[b] = np.append(self._greedy_layer_bias[b], self.xavier_weight_initialization(n_out, n_in, shape=(n_out, 1)))

    def grow_layer(self, option, number_of_nodes):
        raise TypeError('Not implemented')

    def prune_greedy_layer_bias(self, layer_number, node_number):
        def remove_nth_element(greedy_bias_tensor, n):
            bias_tensor = torch.cat([greedy_bias_tensor[0][:n], greedy_bias_tensor[0][n + 1:]])
            return bias_tensor.view(1, bias_tensor.shape[0])

        b = layer_number  # readability
        n = node_number  # readability

        if b is self.number_hidden_layers:
            self._greedy_layer_output_bias = remove_nth_element(self._greedy_layer_output_bias, n)
        else:
            self._greedy_layer_bias[b] = remove_nth_element(self._greedy_layer_bias[b], n)

    def greedy_layer_wise_pretrain(self, x: torch.tensor, number_epochs: int = 1, is_tied_weight: bool = False,
                                   noise_ratio: float = 0.0):
        for i in range(len(self.layers) - 1):
            if i > self.number_hidden_layers:
                nn = NeuralNetwork([self.layers[i], self.layers[-1], self.layers[i]], init_weights=False)
            else:
                nn = NeuralNetwork([self.layers[i], self.layers[i + 1], self.layers[i]], init_weights=False)

            nn.activation_function[0] = nn.ACTIVATION_FUNCTION_SIGMOID
            nn.output_activation_function = nn.ACTIVATION_FUNCTION_SIGMOID
            nn.loss_function = nn.LOSS_FUNCTION_MSE
            nn.momentum_rate = 0

            if i >= self.number_hidden_layers:
                nn.weight[0] = self.output_weight.clone()
                nn.bias[0] = self.output_bias.clone()
                nn.output_weight = self.output_weight.T.clone()
                if self._greedy_layer_output_bias is None:
                    nodes_after = nn.layers[-1]

                    self._greedy_layer_output_bias = self.xavier_weight_initialization(1, nodes_after)
                nn.output_bias = self._greedy_layer_output_bias.clone()
            else:
                nn.weight[0] = self.weight[i].clone()
                nn.bias[0] = self.bias[i].clone()
                nn.output_weight = self.weight[i].T.clone()
                try:
                    nn.output_bias = self._greedy_layer_bias[i].detach()
                except (TypeError, IndexError):
                    nodes_after = nn.layers[-1]

                    if self._greedy_layer_bias is None:
                        self._greedy_layer_bias = []

                    self._greedy_layer_bias.append(self.xavier_weight_initialization(1, nodes_after))
                    nn.output_bias = self._greedy_layer_bias[i].clone()

            for j in range(0, number_epochs):
                training_x = self.forward_pass(x=x).layer_value[i].detach()
                nn.train(x=self.masking_noise(x=training_x, noise_ratio=noise_ratio), y=training_x)

            if i >= self.number_hidden_layers:
                self.output_weight = nn.weight[0].clone()
                self.output_bias = nn.bias[0].clone()
            else:
                self.weight[i] = nn.weight[0].clone()
                self.bias[i] = nn.bias[0].clone()

    def update_weights_kullback_leibler(self, Xs, Xt, gamma=0.0001):
        loss = NeuralNetwork.update_weights_kullback_leibler(self, Xs, Xs, Xt, Xt, gamma)
        return loss

    def compute_evaluation_window(self, x):
        raise TypeError('Not implemented')

    def compute_bias(self, y):
        return torch.mean((self.Ey.T - y) ** 2)

    @property
    def network_variance(self):
        return torch.mean(self.Ey2 - self.Ey ** 2)


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, layers=[]):
        AutoEncoder.__init__(self, layers)
        # FIXME: The lines below are just to build the greedy_layer_bias. Find a more intuitive way to perform it
        random_x = np.random.rand(layers[0])
        random_x = torch.tensor(np.atleast_2d(random_x), dtype=torch.float, device=MyDevice().get())
        self.greedy_layer_wise_pretrain(x=random_x, number_epochs=0)

    def train(self, x: torch.tensor, noise_ratio: float = 0.0, is_tied_weight: bool = False, weight_number: int = None, y: torch.tensor = None):
        AutoEncoder.train(self, x=x, noise_ratio=noise_ratio, is_tied_weight=is_tied_weight, weight_number=weight_number, y=y)

    def greedy_layer_wise_pretrain(self, x: torch.tensor, number_epochs: int = 1, is_tied_weight: bool = False, noise_ratio: float = 0.0, y: torch.tensor = None):
        AutoEncoder.greedy_layer_wise_pretrain(self, x=x, number_epochs=number_epochs, is_tied_weight=is_tied_weight, noise_ratio=noise_ratio)