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

from ACDCDataManipulator import DataManipulator
from NeuralNetwork import NeuralNetwork
from AutoEncoder import DenoisingAutoEncoder
from MySingletons import MyDevice
from colorama import Fore, Back, Style
from itertools import cycle
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pylab as plt
import math
import torch
import time
import random


def __copy_weights(source: NeuralNetwork, targets: list, layer_numbers=None, copy_moment: bool = True):
    if layer_numbers is None:
        layer_numbers = [1]
    if type(targets) is not list:
        targets = [targets]
    for layer_number in layer_numbers:
        layer_number -= 1
        for target in targets:
            if layer_number >= source.number_hidden_layers:
                target.output_weight = source.output_weight.detach()
                target.output_bias = source.output_bias.detach()
                if copy_moment:
                    target.output_momentum = source.output_momentum.detach()
                    target.output_bias_momentum = source.output_bias_momentum.detach()
            else:
                target.weight[layer_number] = source.weight[layer_number].detach()
                target.bias[layer_number] = source.bias[layer_number].detach()
                if copy_moment:
                    target.momentum[layer_number] = source.momentum[layer_number].detach()
                    target.bias_momentum[layer_number] = source.bias_momentum[layer_number].detach()


def __grow_nodes(*networks):
    origin = networks[0]
    if origin.growable[origin.number_hidden_layers]:
        nodes = 1
        for i in range(nodes):
            for network in networks:
                network.grow_node(origin.number_hidden_layers)
        return True
    else:
        return False


def __prune_nodes(*networks):
    origin = networks[0]
    if origin.prunable[origin.number_hidden_layers][0] >= 0:
        nodes_to_prune = origin.prunable[origin.number_hidden_layers].tolist()
        for network in networks:
            for node_to_prune in nodes_to_prune[::-1]:
                network.prune_node(origin.number_hidden_layers, node_to_prune)
        return True
    return False


def __width_evolution(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, is_gpu=False):
    if y is None:
        y = x

    network.feedforward(x, y)
    network.width_adaptation_stepwise(y, is_gpu=is_gpu)


def __discriminative(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None, is_neg_grad: bool = False):
    y = x.detach() if y is None else y
    network.train(x=x, y=y, is_neg_grad=is_neg_grad)


def __generative(network: DenoisingAutoEncoder, x: torch.tensor, y: torch.tensor = None,
                 is_tied_weight=True, noise_ratio=0.1, glw_epochs: int = 1):
    y = x.detach() if y is None else y
    network.greedy_layer_wise_pretrain(x=x, number_epochs=glw_epochs, noise_ratio=noise_ratio)
    network.train(x=x, y=y, noise_ratio=noise_ratio, is_tied_weight=is_tied_weight)


def __test(network: NeuralNetwork, x: torch.tensor, y: torch.tensor = None,
           is_source: bool = False, is_discriminative: bool = False, metrics=None):
    with torch.no_grad():
        y = x.detach() if y is None else y
        network.test(x=x, y=y)

        if is_source:
            if is_discriminative:
                metrics['y_true_source'].extend(network.true_classes.tolist())
                metrics['y_pred_source'].extend(network.outputed_classes.tolist())
                metrics['f1_score_source_micro'].append(f1_score(metrics['y_true_source'], metrics['y_pred_source'], average='micro'))
                metrics['f1_score_source_macro'].append(f1_score(metrics['y_true_source'], metrics['y_pred_source'], average='macro'))
                metrics['f1_score_source_weighted'].append(f1_score(metrics['y_true_source'], metrics['y_pred_source'], average='weighted'))
                metrics['classification_rate_source'].append(network.classification_rate)
                metrics['classification_source_loss'].append(float(network.loss_value))
                metrics['classification_source_misclassified'].append(float(network.misclassified))
            else:
                metrics['reconstruction_source_loss'].append(float(network.loss_value))
        else:
            if is_discriminative:
                metrics['y_true_target'].extend(network.true_classes.tolist())
                metrics['y_pred_target'].extend(network.outputed_classes.tolist())
                metrics['f1_score_target_micro'].append(f1_score(metrics['y_true_target'], metrics['y_pred_target'], average='micro'))
                metrics['f1_score_target_macro'].append(f1_score(metrics['y_true_target'], metrics['y_pred_target'], average='macro'))
                metrics['f1_score_target_weighted'].append(f1_score(metrics['y_true_target'], metrics['y_pred_target'], average='weighted'))
                metrics['classification_rate_target'].append(network.classification_rate)
                metrics['classification_target_loss'].append(float(network.loss_value))
                metrics['classification_target_misclassified'].append(float(network.misclassified))
            else:
                metrics['reconstruction_target_loss'].append(float(network.loss_value))


def __force_same_size(a_tensor, b_tensor, shuffle=True, strategy='max'):
    common = np.min([a_tensor.shape[0], b_tensor.shape[0]])

    if shuffle:
        a_tensor = a_tensor[torch.randperm(a_tensor.shape[0])]
        b_tensor = b_tensor[torch.randperm(b_tensor.shape[0])]

    if strategy == 'max':
        if math.ceil(a_tensor.shape[0] / common) <= math.ceil(b_tensor.shape[0] / common):
            b_tensor = torch.stack(list(target for target, source in zip(b_tensor, cycle(a_tensor))))
            a_tensor = torch.stack(list(source for target, source in zip(b_tensor, cycle(a_tensor))))
        else:
            b_tensor = torch.stack(list(target for target, source in zip(cycle(b_tensor), a_tensor)))
            a_tensor = torch.stack(list(source for target, source in zip(cycle(b_tensor), a_tensor)))

    elif strategy == 'min':
        a_tensor = a_tensor[:common]
        b_tensor = b_tensor[:common]

    if shuffle:
        a_tensor = a_tensor[torch.randperm(a_tensor.shape[0])]
        b_tensor = b_tensor[torch.randperm(b_tensor.shape[0])]

    return a_tensor, b_tensor


def __print_annotation(lst):
    def custom_range(xx):
        step = int(len(xx) * 0.25) - 1
        return range(0, len(xx), 1 if step == 0 else step)

    for idx in custom_range(lst):
        pos = lst[idx] if isinstance(lst[idx], (int, float, np.int32)) else lst[idx][0]
        plt.annotate(format(pos, '.2f'), (idx, pos))
    pos = lst[-1] if isinstance(lst[-1], (int, float, np.int32)) else lst[-1][0]
    plt.annotate(format(pos, '.2f'), (len(lst), pos))


def __plot_time(train_time: np.ndarray,
                test_time: np.ndarray,
                annotation=True):
    plt.title('Processing time')
    plt.ylabel('Seconds')
    plt.xlabel('Minibatches')

    plt.plot(train_time, linewidth=1,
             label=('Train time: %f (Mean) %f (Accumulated)' %
                    (np.nanmean(train_time), np.sum(train_time))))
    plt.plot(test_time, linewidth=1,
             label=('Test time: %f (Mean) %f (Accumulated)' %
                    (np.nanmean(test_time), np.sum(test_time))))
    plt.legend()

    if annotation:
        __print_annotation(train_time)
        __print_annotation(test_time)

    plt.tight_layout()
    plt.show()


def __plot_node_evolution(nodes_discriminator: np.ndarray,
                          nodes_domain_classifier: np.ndarray,
                          nodes_feature_extraction: np.ndarray,
                          annotation=True):
    plt.title('Node evolution')
    plt.ylabel('Nodes')
    plt.xlabel('Minibatches')

    plt.plot(nodes_discriminator, linewidth=1,
             label=('Discriminator HL nodes: %f (Mean) %d (Final)' %
                    (np.nanmean(nodes_discriminator), nodes_discriminator[-1])))
    plt.plot(nodes_domain_classifier, linewidth=1,
             label=('Domain Classifier HL nodes: %f (Mean) %d (Final)' %
                    (np.nanmean(nodes_domain_classifier), nodes_domain_classifier[-1])))
    plt.plot(nodes_feature_extraction, linewidth=1,
             label=('Feature Extraction HL nodes: %f (Mean) %d (Final)' %
                    (np.nanmean(nodes_feature_extraction), nodes_feature_extraction[-1])))
    plt.legend()

    if annotation:
        __print_annotation(nodes_discriminator)
        __print_annotation(nodes_domain_classifier)
        __print_annotation(nodes_feature_extraction)

    plt.tight_layout()
    plt.show()


def __plot_losses(classification_source_loss: np.ndarray,
                  classification_target_loss: np.ndarray,
                  reconstruction_source_loss: np.ndarray,
                  reconstruction_target_loss: np.ndarray,
                  domain_classifier_loss: np.ndarray,
                  annotation=True):
    plt.title('Losses evolution')
    plt.ylabel('Loss value')
    plt.xlabel('Minibatches')

    plt.plot(classification_source_loss, linewidth=1,
             label=('Classification Source Loss mean: %f' %
                    (np.nanmean(classification_source_loss))))
    plt.plot(classification_target_loss, linewidth=1,
             label=('Classification Target Loss mean: %f' %
                    (np.nanmean(classification_target_loss))))
    plt.plot(reconstruction_source_loss, linewidth=1,
             label=('Reconstruction Source Loss mean: %f' %
                    (np.nanmean(reconstruction_source_loss))))
    plt.plot(reconstruction_target_loss, linewidth=1,
             label=('Reconstruction Target Loss mean: %f' %
                    (np.nanmean(reconstruction_target_loss))))
    plt.plot(domain_classifier_loss, linewidth=1,
             label=('Domain Classifier Loss mean: %f' %
                    (np.nanmean(domain_classifier_loss))))
    plt.legend()

    if annotation:
        __print_annotation(classification_source_loss)
        __print_annotation(classification_target_loss)
        __print_annotation(reconstruction_source_loss)
        __print_annotation(reconstruction_target_loss)
        __print_annotation(domain_classifier_loss)

    plt.tight_layout()
    plt.show()


def __plot_classification_rates(source_rate: np.ndarray,
                                target_rate: np.ndarray,
                                domain_rate: np.ndarray,
                                total_source_rate: float,
                                total_target_rate: float,
                                total_domain_classification_rate: float,
                                annotation=True,
                                class_number=None):
    plt.title('Source and Target Classification Rates')
    plt.ylabel('Classification Rate')
    plt.xlabel('Minibatches')

    plt.plot(source_rate, linewidth=1, label=('Source CR: %f (batch) | %f (dataset)' %
                                              (np.nanmean(source_rate), total_source_rate)))
    plt.plot(target_rate, linewidth=1, label=('Target CR: %f (batch) | %f (dataset)' %
                                              (np.nanmean(target_rate), total_target_rate)))
    plt.plot(domain_rate, linewidth=1, label=('Domain CR: %f (batch) | %f (dataset)' %
                                              (np.nanmean(domain_rate), total_domain_classification_rate)))

    if annotation:
        __print_annotation(source_rate)
        __print_annotation(target_rate)
        __print_annotation(domain_rate)

    if class_number is not None:
        plt.plot(np.ones(len(source_rate)) * 1 / class_number,
                 linewidth=1, label='Random Classification Threshold: %f' % (1 / class_number))

    plt.plot(np.ones(len(source_rate)) * 1 / 2,
             linewidth=1, label='Random Domain Classification Threshold: %f' % (1 / 2))

    plt.legend()

    plt.tight_layout()
    plt.show()


def __plot_ns(bias, var, ns, annotation=True):
    plt.plot(bias, linewidth=1, label=('Bias mean: %f' % (np.nanmean(bias))))
    plt.plot(var, linewidth=1, label=('Variance mean: %f' % (np.nanmean(var))))
    plt.plot(ns, linewidth=1, label=('NS (Bias + Variance) mean: %f' % (np.nanmean(ns))))
    plt.legend()

    if annotation:
        __print_annotation(bias)
        __print_annotation(var)
        __print_annotation(ns)

    plt.tight_layout()
    plt.show()


def __plot_discriminative_network_significance(bias, var, annotation=True):
    plt.title('Discriminative Network Significance')
    plt.ylabel('Value')
    plt.xlabel('Sample')

    __plot_ns(bias, var, (np.array(bias) + np.array(var)).tolist(), annotation)


def __plot_domain_classifier_network_significance(bias, var, annotation=True):
    plt.title('Domain Classifier Network Significance')
    plt.ylabel('Value')
    plt.xlabel('Sample')

    __plot_ns(bias, var, (np.array(bias) + np.array(var)).tolist(), annotation)


def __plot_feature_extractor_network_significance(bias, var, annotation=True):
    plt.title('Feature Extractor Network Significance')
    plt.ylabel('Value')
    plt.xlabel('Sample')

    __plot_ns(bias, var, (np.array(bias) + np.array(var)).tolist(), annotation)


def __load_source_target(source: str, target: str, n_source_concept_drift: int = 1, n_target_concept_drift: int = 1):
    dm_s = DataManipulator()
    dm_t = DataManipulator()

    source = source.replace('_', '-').replace(' ', '-').lower()
    target = target.replace('_', '-').replace(' ', '-').lower()

    if source == 'mnist-28':
        dm_s.load_mnist(resize=28, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-26':
        dm_s.load_mnist(resize=26, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-24':
        dm_s.load_mnist(resize=24, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-22':
        dm_s.load_mnist(resize=22, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-20':
        dm_s.load_mnist(resize=20, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-18':
        dm_s.load_mnist(resize=18, n_concept_drifts=n_source_concept_drift)
    elif source == 'mnist-16':
        dm_s.load_mnist(resize=16, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-28':
        dm_s.load_usps(resize=28, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-26':
        dm_s.load_usps(resize=26, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-24':
        dm_s.load_usps(resize=24, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-22':
        dm_s.load_usps(resize=22, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-20':
        dm_s.load_usps(resize=20, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-18':
        dm_s.load_usps(resize=18, n_concept_drifts=n_source_concept_drift)
    elif source == 'usps-16':
        dm_s.load_usps(resize=16, n_concept_drifts=n_source_concept_drift)
    elif source == 'cifar10':
        dm_s.load_cifar10(n_concept_drifts=n_source_concept_drift)
    elif source == 'stl10':
        dm_s.load_stl10(n_concept_drifts=n_source_concept_drift)
    elif source == 'london-bike':
        dm_s.load_london_bike_sharing(n_concept_drifts=n_source_concept_drift)
    elif source == 'washington-bike':
        dm_s.load_washington_bike_sharing(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-fashion':
        dm_s.load_amazon_review_fashion(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-all-beauty':
        dm_s.load_amazon_review_all_beauty(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-appliances':
        dm_s.load_amazon_review_appliances(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-arts-crafts-sewing':
        dm_s.load_amazon_review_arts_crafts_sewing(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-automotive':
        dm_s.load_amazon_review_automotive(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-books':
        dm_s.load_amazon_review_books(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-cds-vinyl':
        dm_s.load_amazon_review_cds_vinyl(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-cellphones_accessories':
        dm_s.load_amazon_review_cellphones_accessories(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-clothing-shoes-jewelry':
        dm_s.load_amazon_review_clothing_shoes_jewelry(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-digital-music':
        dm_s.load_amazon_review_digital_music(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-electronics':
        dm_s.load_amazon_review_electronics(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-gift-card':
        dm_s.load_amazon_review_gift_card(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-grocery-gourmet-food':
        dm_s.load_amazon_review_grocery_gourmet_food(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-home-kitchen':
        dm_s.load_amazon_review_home_kitchen(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-industrial-scientific':
        dm_s.load_amazon_review_industrial_scientific(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-kindle-store':
        dm_s.load_amazon_review_kindle_store(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-luxury-beauty':
        dm_s.load_amazon_review_luxury_beauty(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-magazine-subscription':
        dm_s.load_amazon_review_magazine_subscription(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-movies-tv':
        dm_s.load_amazon_review_movies_tv(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-musical-instruments':
        dm_s.load_amazon_review_musical_instruments(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-office-products':
        dm_s.load_amazon_review_office_products(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-patio-lawn-garden':
        dm_s.load_amazon_review_patio_lawn_garden(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-pet-supplies':
        dm_s.load_amazon_review_pet_supplies(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-prime-pantry':
        dm_s.load_amazon_review_prime_pantry(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-software':
        dm_s.load_amazon_review_software(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-sports-outdoors':
        dm_s.load_amazon_review_sports_outdoors(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-tools-home-improvements':
        dm_s.load_amazon_review_tools_home_improvements(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-toys-games':
        dm_s.load_amazon_review_toys_games(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-video-games':
        dm_s.load_amazon_review_video_games(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-nips-books':
        dm_s.load_amazon_review_nips_books(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-nips-dvd':
        dm_s.load_amazon_review_nips_dvd(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-nips-electronics':
        dm_s.load_amazon_review_nips_electronics(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-nips-kitchen':
        dm_s.load_amazon_review_nips_kitchen(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-apparel':
        dm_s.load_amazon_review_acl_apparel(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-automotive':
        dm_s.load_amazon_review_acl_automotive(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-baby':
        dm_s.load_amazon_review_acl_baby(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-beauty':
        dm_s.load_amazon_review_acl_beauty(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-books':
        dm_s.load_amazon_review_acl_books(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-camera_photo':
        dm_s.load_amazon_review_acl_camera_photo(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-cell_phones_service':
        dm_s.load_amazon_review_acl_cell_phones_service(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-computer_video_games':
        dm_s.load_amazon_review_acl_computer_video_games(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-dvd':
        dm_s.load_amazon_review_acl_dvd(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-electronics':
        dm_s.load_amazon_review_acl_electronics(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-gourmet_food':
        dm_s.load_amazon_review_acl_gourmet_food(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-grocery':
        dm_s.load_amazon_review_acl_grocery(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-health_personal_care':
        dm_s.load_amazon_review_acl_health_personal_care(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-jewelry_watches':
        dm_s.load_amazon_review_acl_jewelry_watches(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-kitchen_housewares':
        dm_s.load_amazon_review_acl_kitchen_housewares(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-magazines':
        dm_s.load_amazon_review_acl_magazines(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-music':
        dm_s.load_amazon_review_acl_music(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-musical_instruments':
        dm_s.load_amazon_review_acl_musical_instruments(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-office_products':
        dm_s.load_amazon_review_acl_office_products(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-outdoor_living':
        dm_s.load_amazon_review_acl_outdoor_living(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-software':
        dm_s.load_amazon_review_acl_software(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-sports_outdoors':
        dm_s.load_amazon_review_acl_sports_outdoors(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-tools_hardware':
        dm_s.load_amazon_review_acl_tools_hardware(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-toys_games':
        dm_s.load_amazon_review_acl_toys_games(n_concept_drifts=n_source_concept_drift)
    elif source == 'amazon-review-acl-video':
        dm_s.load_amazon_review_acl_video(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-obama-all':
        dm_s.load_news_popularity_obama_all(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-economy-all':
        dm_s.load_news_popularity_economy_all(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-microsoft-all':
        dm_s.load_news_popularity_microsoft_all(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-palestine-all':
        dm_s.load_news_popularity_palestine_all(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-obama-facebook':
        dm_s.load_news_popularity_obama_facebook(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-economy-facebook':
        dm_s.load_news_popularity_economy_facebook(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-microsoft-facebook':
        dm_s.load_news_popularity_microsoft_facebook(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-palestine-facebook':
        dm_s.load_news_popularity_palestine_facebook(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-obama-googleplus':
        dm_s.load_news_popularity_obama_googleplus(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-economy-googleplus':
        dm_s.load_news_popularity_economy_googleplus(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-microsoft-googleplus':
        dm_s.load_news_popularity_microsoft_googleplus(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-palestine-googleplus':
        dm_s.load_news_popularity_palestine_googleplus(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-obama-linkedin':
        dm_s.load_news_popularity_obama_linkedin(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-economy-linkedin':
        dm_s.load_news_popularity_economy_linkedin(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-microsoft-linkedin':
        dm_s.load_news_popularity_microsoft_linkedin(n_concept_drifts=n_source_concept_drift)
    elif source == 'news-palestine-linkedin':
        dm_s.load_news_popularity_palestine_linkedin(n_concept_drifts=n_source_concept_drift)

    if target == 'mnist-28':
        dm_t.load_mnist(resize=28, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-26':
        dm_t.load_mnist(resize=26, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-24':
        dm_t.load_mnist(resize=24, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-22':
        dm_t.load_mnist(resize=22, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-20':
        dm_t.load_mnist(resize=20, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-18':
        dm_t.load_mnist(resize=18, n_concept_drifts=n_target_concept_drift)
    elif target == 'mnist-16':
        dm_t.load_mnist(resize=16, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-28':
        dm_t.load_usps(resize=28, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-26':
        dm_t.load_usps(resize=26, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-24':
        dm_t.load_usps(resize=24, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-22':
        dm_t.load_usps(resize=22, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-20':
        dm_t.load_usps(resize=20, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-18':
        dm_t.load_usps(resize=18, n_concept_drifts=n_target_concept_drift)
    elif target == 'usps-16':
        dm_t.load_usps(resize=16, n_concept_drifts=n_target_concept_drift)
    elif target == 'cifar10':
        dm_t.load_cifar10(n_concept_drifts=n_target_concept_drift)
    elif target == 'stl10':
        dm_t.load_stl10(n_concept_drifts=n_target_concept_drift)
    elif target == 'london-bike':
        dm_t.load_london_bike_sharing(n_concept_drifts=n_source_concept_drift)
    elif target == 'washington-bike':
        dm_t.load_washington_bike_sharing(n_concept_drifts=n_source_concept_drift)
    elif target == 'amazon-review-fashion':
        dm_t.load_amazon_review_fashion(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-all-beauty':
        dm_t.load_amazon_review_all_beauty(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-appliances':
        dm_t.load_amazon_review_appliances(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-arts-crafts-sewing':
        dm_t.load_amazon_review_arts_crafts_sewing(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-automotive':
        dm_t.load_amazon_review_automotive(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-books':
        dm_t.load_amazon_review_books(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-cds-vinyl':
        dm_t.load_amazon_review_cds_vinyl(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-cellphones_accessories':
        dm_t.load_amazon_review_cellphones_accessories(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-clothing-shoes-jewelry':
        dm_t.load_amazon_review_clothing_shoes_jewelry(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-digital-music':
        dm_t.load_amazon_review_digital_music(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-electronics':
        dm_t.load_amazon_review_electronics(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-gift-card':
        dm_t.load_amazon_review_gift_card(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-grocery-gourmet-food':
        dm_t.load_amazon_review_grocery_gourmet_food(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-home-kitchen':
        dm_t.load_amazon_review_home_kitchen(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-industrial-scientific':
        dm_t.load_amazon_review_industrial_scientific(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-kindle-store':
        dm_t.load_amazon_review_kindle_store(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-luxury-beauty':
        dm_t.load_amazon_review_luxury_beauty(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-magazine-subscription':
        dm_t.load_amazon_review_magazine_subscription(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-movies-tv':
        dm_t.load_amazon_review_movies_tv(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-musical-instruments':
        dm_t.load_amazon_review_musical_instruments(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-office-products':
        dm_t.load_amazon_review_office_products(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-patio-lawn-garden':
        dm_t.load_amazon_review_patio_lawn_garden(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-pet-supplies':
        dm_t.load_amazon_review_pet_supplies(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-prime-pantry':
        dm_t.load_amazon_review_prime_pantry(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-software':
        dm_t.load_amazon_review_software(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-sports-outdoors':
        dm_t.load_amazon_review_sports_outdoors(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-tools-home-improvements':
        dm_t.load_amazon_review_tools_home_improvements(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-toys-games':
        dm_t.load_amazon_review_toys_games(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-video-games':
        dm_t.load_amazon_review_video_games(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-nips-books':
        dm_t.load_amazon_review_nips_books(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-nips-dvd':
        dm_t.load_amazon_review_nips_dvd(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-nips-electronics':
        dm_t.load_amazon_review_nips_electronics(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-nips-kitchen':
        dm_t.load_amazon_review_nips_kitchen(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-apparel':
        dm_t.load_amazon_review_acl_apparel(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-automotive':
        dm_t.load_amazon_review_acl_automotive(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-baby':
        dm_t.load_amazon_review_acl_baby(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-beauty':
        dm_t.load_amazon_review_acl_beauty(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-books':
        dm_t.load_amazon_review_acl_books(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-camera_photo':
        dm_t.load_amazon_review_acl_camera_photo(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-cell_phones_service':
        dm_t.load_amazon_review_acl_cell_phones_service(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-computer_video_games':
        dm_t.load_amazon_review_acl_computer_video_games(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-dvd':
        dm_t.load_amazon_review_acl_dvd(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-electronics':
        dm_t.load_amazon_review_acl_electronics(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-gourmet_food':
        dm_t.load_amazon_review_acl_gourmet_food(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-grocery':
        dm_t.load_amazon_review_acl_grocery(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-health_personal_care':
        dm_t.load_amazon_review_acl_health_personal_care(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-jewelry_watches':
        dm_t.load_amazon_review_acl_jewelry_watches(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-kitchen_housewares':
        dm_t.load_amazon_review_acl_kitchen_housewares(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-magazines':
        dm_t.load_amazon_review_acl_magazines(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-music':
        dm_t.load_amazon_review_acl_music(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-musical_instruments':
        dm_t.load_amazon_review_acl_musical_instruments(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-office_products':
        dm_t.load_amazon_review_acl_office_products(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-outdoor_living':
        dm_t.load_amazon_review_acl_outdoor_living(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-software':
        dm_t.load_amazon_review_acl_software(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-sports_outdoors':
        dm_t.load_amazon_review_acl_sports_outdoors(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-tools_hardware':
        dm_t.load_amazon_review_acl_tools_hardware(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-toys_games':
        dm_t.load_amazon_review_acl_toys_games(n_concept_drifts=n_target_concept_drift)
    elif target == 'amazon-review-acl-video':
        dm_t.load_amazon_review_acl_video(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-obama-all':
        dm_t.load_news_popularity_obama_all(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-economy-all':
        dm_t.load_news_popularity_economy_all(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-microsoft-all':
        dm_t.load_news_popularity_microsoft_all(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-palestine-all':
        dm_t.load_news_popularity_palestine_all(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-obama-facebook':
        dm_t.load_news_popularity_obama_facebook(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-economy-facebook':
        dm_t.load_news_popularity_economy_facebook(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-microsoft-facebook':
        dm_t.load_news_popularity_microsoft_facebook(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-palestine-facebook':
        dm_t.load_news_popularity_palestine_facebook(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-obama-googleplus':
        dm_t.load_news_popularity_obama_googleplus(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-economy-googleplus':
        dm_t.load_news_popularity_economy_googleplus(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-microsoft-googleplus':
        dm_t.load_news_popularity_microsoft_googleplus(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-palestine-googleplus':
        dm_t.load_news_popularity_palestine_googleplus(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-obama-linkedin':
        dm_t.load_news_popularity_obama_linkedin(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-economy-linkedin':
        dm_t.load_news_popularity_economy_linkedin(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-microsoft-linkedin':
        dm_t.load_news_popularity_microsoft_linkedin(n_concept_drifts=n_target_concept_drift)
    elif target == 'news-palestine-linkedin':
        dm_t.load_news_popularity_palestine_linkedin(n_concept_drifts=n_target_concept_drift)

    return dm_s, dm_t


def acdc(source, target,
         n_source_concept_drift: int = 5,
         n_target_concept_drift: int = 7,
         internal_epochs: int = 1, is_gpu=True):
    def print_metrics(minibatch, metrics, DMs, DMt, NN, DAEt, DA):
        print('Minibatch: %d | Execution time (dataset load/pre-processing + model run): %f' % (
            minibatch, time.time() - metrics['start_execution_time']))
        if minibatch > 1:
            print((
                          'Total of samples:' + Fore.BLUE + ' %d + %d = %d/%d (%.2f%%) Source' + Style.RESET_ALL + ' |' + Fore.RED + ' %d + %d = %d/%d (%.2f%%) Target' + Style.RESET_ALL + ' | %d/%d (%.2f%%) Samples in total') % (
                      metrics['number_evaluated_samples_source'][-2],
                      metrics['number_evaluated_samples_source'][-1] - metrics['number_evaluated_samples_source'][-2],
                      metrics['number_evaluated_samples_source'][-1],
                      DMs.number_samples(),
                      float(metrics['number_evaluated_samples_source'][-1] / DMs.number_samples()) * 100,
                      metrics['number_evaluated_samples_target'][-2],
                      metrics['number_evaluated_samples_target'][-1] - metrics['number_evaluated_samples_target'][-2],
                      metrics['number_evaluated_samples_target'][-1],
                      DMt.number_samples(),
                      float(metrics['number_evaluated_samples_target'][-1] / DMt.number_samples()) * 100,
                      metrics['number_evaluated_samples_source'][-1] + metrics['number_evaluated_samples_target'][-1],
                      DMs.number_samples() + DMt.number_samples(),
                      float((metrics['number_evaluated_samples_source'][-1] +
                             metrics['number_evaluated_samples_target'][-1]) / (
                                    DMs.number_samples() + DMt.number_samples())) * 100))
        else:
            print((
                          'Total of samples:' + Fore.BLUE + ' %d/%d (%.2f%%) Source' + Style.RESET_ALL + ' |' + Fore.RED + ' %d/%d (%.2f%%) Target' + Style.RESET_ALL + ' | %d/%d (%.2f%%) Samples in total') % (
                      metrics['number_evaluated_samples_source'][-1],
                      DMs.number_samples(),
                      float(metrics['number_evaluated_samples_source'][-1] / DMs.number_samples()) * 100,
                      metrics['number_evaluated_samples_target'][-1],
                      DMt.number_samples(),
                      float(metrics['number_evaluated_samples_target'][-1] / DMt.number_samples()) * 100,
                      metrics['number_evaluated_samples_source'][-1] + metrics['number_evaluated_samples_target'][-1],
                      DMs.number_samples() + DMt.number_samples(),
                      float((metrics['number_evaluated_samples_source'][-1] +
                             metrics['number_evaluated_samples_target'][-1]) / (
                                    DMs.number_samples() + DMt.number_samples())) * 100))

        if minibatch > 1:
            string_max = '' + Fore.GREEN + 'Max' + Style.RESET_ALL
            string_mean = '' + Fore.YELLOW + 'Mean' + Style.RESET_ALL
            string_min = '' + Fore.RED + 'Min' + Style.RESET_ALL
            string_now = '' + Fore.BLUE + 'Now' + Style.RESET_ALL
            string_accu = '' + Fore.MAGENTA + 'Accu' + Style.RESET_ALL

            print((
                          '%s %s %s %s %s Training time:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Fore.MAGENTA + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now, string_accu,
                      np.max(metrics['train_time']),
                      np.nanmean(metrics['train_time']),
                      np.min(metrics['train_time']),
                      metrics['train_time'][-1],
                      np.sum(metrics['train_time'])))
            print((
                          '%s %s %s %s %s Testing time:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Fore.MAGENTA + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now, string_accu,
                      np.max(metrics['test_time']),
                      np.nanmean(metrics['test_time']),
                      np.min(metrics['test_time']),
                      metrics['test_time'][-1],
                      np.sum(metrics['test_time'])))
            print((
                          '%s %s %s %s CR Source:' + Fore.GREEN + ' %f%% ' + Back.BLUE + Fore.YELLOW + Style.BRIGHT + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_rate_source']) * 100,
                      np.nanmean(metrics['classification_rate_source']) * 100,
                      np.min(metrics['classification_rate_source']) * 100,
                      metrics['classification_rate_source'][-1] * 100))
            print((
                          '%s %s %s %s CR Target:' + Fore.GREEN + ' %f%% ' + Back.RED + Fore.YELLOW + Style.BRIGHT + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_rate_target']) * 100,
                      np.nanmean(metrics['classification_rate_target']) * 100,
                      np.min(metrics['classification_rate_target']) * 100,
                      metrics['classification_rate_target'][-1] * 100))
            print((
                          '%s %s %s %s CR Domain Discriminator:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_rate_domain']) * 100,
                      np.nanmean(metrics['classification_rate_domain']) * 100,
                      np.min(metrics['classification_rate_domain']) * 100,
                      metrics['classification_rate_domain'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Micro Source:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_source_micro']) * 100,
                      np.nanmean(metrics['f1_score_source_micro']) * 100,
                      np.min(metrics['f1_score_source_micro']) * 100,
                      metrics['f1_score_source_micro'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Micro Target:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_target_micro']) * 100,
                      np.nanmean(metrics['f1_score_target_micro']) * 100,
                      np.min(metrics['f1_score_target_micro']) * 100,
                      metrics['f1_score_target_micro'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Macro Source:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_source_macro']) * 100,
                      np.nanmean(metrics['f1_score_source_macro']) * 100,
                      np.min(metrics['f1_score_source_macro']) * 100,
                      metrics['f1_score_source_macro'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Macro Target:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_target_macro']) * 100,
                      np.nanmean(metrics['f1_score_target_macro']) * 100,
                      np.min(metrics['f1_score_target_macro']) * 100,
                      metrics['f1_score_target_macro'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Weighted Source:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_source_weighted']) * 100,
                      np.nanmean(metrics['f1_score_source_weighted']) * 100,
                      np.min(metrics['f1_score_source_weighted']) * 100,
                      metrics['f1_score_source_weighted'][-1] * 100))
            print((
                          '%s %s %s %s F1 Score Weighted Target:' + Fore.GREEN + ' %f%% ' + Fore.YELLOW + '%f%%' + Style.RESET_ALL + Fore.RED + ' %f%%' + Fore.BLUE + ' %f%%' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['f1_score_target_weighted']) * 100,
                      np.nanmean(metrics['f1_score_target_weighted']) * 100,
                      np.min(metrics['f1_score_target_weighted']) * 100,
                      metrics['f1_score_target_weighted'][-1] * 100))
            print((
                          '%s %s %s %s Classification Source Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_source_loss']),
                      np.nanmean(metrics['classification_source_loss']),
                      np.min(metrics['classification_source_loss']),
                      metrics['classification_source_loss'][-1]))
            print((
                          '%s %s %s %s Classification Target Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['classification_target_loss']),
                      np.nanmean(metrics['classification_target_loss']),
                      np.min(metrics['classification_target_loss']),
                      metrics['classification_target_loss'][-1]))
            print((
                          '%s %s %s %s Domain Discriminator Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['domain_regression_loss']),
                      np.nanmean(metrics['domain_regression_loss']),
                      np.min(metrics['domain_regression_loss']),
                      metrics['domain_regression_loss'][-1]))
            print((
                          '%s %s %s %s Reconstruction Source Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['reconstruction_source_loss']),
                      np.nanmean(metrics['reconstruction_source_loss']),
                      np.min(metrics['reconstruction_source_loss']),
                      metrics['reconstruction_source_loss'][-1]))
            print((
                          '%s %s %s %s Reconstruction Target Loss:' + Fore.GREEN + ' %f' + Fore.YELLOW + ' %f' + Fore.RED + ' %f' + Fore.BLUE + ' %f' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['reconstruction_target_loss']),
                      np.nanmean(metrics['reconstruction_target_loss']),
                      np.min(metrics['reconstruction_target_loss']),
                      metrics['reconstruction_target_loss'][-1]))
            print((
                          '%s %s %s %s Discriminator Nodes:' + Fore.GREEN + ' %d' + Fore.YELLOW + ' %f' + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['node_evolution_discriminator']),
                      np.nanmean(metrics['node_evolution_discriminator']),
                      np.min(metrics['node_evolution_discriminator']),
                      metrics['node_evolution_discriminator'][-1]))
            print((
                          '%s %s %s %s Denoising Autoencoder Nodes:' + Fore.GREEN + ' %d' + Fore.YELLOW + ' %f' + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['node_evolution_feature_extraction']),
                      np.nanmean(metrics['node_evolution_feature_extraction']),
                      np.min(metrics['node_evolution_feature_extraction']),
                      metrics['node_evolution_feature_extraction'][-1]))
            print((
                          '%s %s %s %s Domain Classifier Nodes:' + Fore.GREEN + ' %d' + Fore.YELLOW + ' %f' + Fore.RED + ' %d' + Fore.BLUE + ' %d' + Style.RESET_ALL) % (
                      string_max, string_mean, string_min, string_now,
                      np.max(metrics['node_evolution_domain_classifier']),
                      np.nanmean(metrics['node_evolution_domain_classifier']),
                      np.min(metrics['node_evolution_domain_classifier']),
                      metrics['node_evolution_domain_classifier'][-1]))
            print(('Network structure:' + Fore.BLUE + ' %s' + Style.RESET_ALL) % (
                " ".join(map(str, NN.layers))))
            print(('Domain Discriminator structure:' + Fore.GREEN + ' %s' + Style.RESET_ALL) % (
                " ".join(map(str, DA.layers))))
            print(('Denoising Auto Encoder:' + Fore.RED + ' %s' + Style.RESET_ALL) % (
                " ".join(map(str, DAEt.layers))))
        print(Style.RESET_ALL)

    metrics = {'classification_rate_source': [],
               'classification_rate_target': [],
               'classification_rate_domain': [],
               'number_evaluated_samples_source': [],
               'number_evaluated_samples_target': [],
               'train_time': [],
               'test_time': [],
               'node_evolution_discriminator': [],
               'node_evolution_domain_classifier': [],
               'node_evolution_feature_extraction': [],
               'classification_target_loss': [],
               'classification_source_loss': [],
               'reconstruction_source_loss': [],
               'reconstruction_target_loss': [],
               'domain_regression_loss': [],
               'classification_source_misclassified': [],
               'classification_target_misclassified': [],
               'domain_classification_misclassified': [],
               'y_true_source': [],
               'y_pred_source': [],
               'y_true_target': [],
               'y_pred_target': [],
               'f1_score_source_micro': [],
               'f1_score_target_micro': [],
               'f1_score_source_macro': [],
               'f1_score_target_macro': [],
               'f1_score_source_weighted': [],
               'f1_score_target_weighted': [],
               'start_execution_time': time.time()}
    MyDevice().set(is_gpu=is_gpu)
    internal_epochs = internal_epochs if internal_epochs >= 1 else 1

    SOURCE_DOMAIN_LABEL = torch.tensor([[1, 0]], dtype=torch.float, device=MyDevice().get())
    TARGET_DOMAIN_LABEL = torch.tensor([[0, 1]], dtype=torch.float, device=MyDevice().get())

    dm_s, dm_t = __load_source_target(source, target, n_source_concept_drift, n_target_concept_drift)

    dae = DenoisingAutoEncoder([dm_s.number_features(),
                                int(dm_s.number_features() * 0.5),
                                dm_s.number_features()])
    nn = NeuralNetwork([dm_s.number_features(),
                        dae.layers[1],
                        1,
                        dm_s.number_classes()])
    da = NeuralNetwork([dm_s.number_features(),
                        dae.layers[1],
                        1,
                        2])

    count_source = 0
    count_target = 0
    count_window = 0
    window_size = 100
    batch_counter = 0

    x_source = []
    y_source = []
    x_target = []
    y_target = []

    while count_source < dm_s.number_samples() \
            or count_target < dm_t.number_samples():
        if count_window < window_size \
                and (count_source < dm_s.number_samples()
                     or count_target < dm_t.number_samples()):

            source_prob = (dm_s.number_samples() - count_source) / (
                    dm_s.number_samples() - count_source + dm_t.number_samples() - count_target + 0.)

            if (np.random.rand() <= source_prob and count_source < dm_s.number_samples()) or (
                    count_target >= dm_t.number_samples() and count_source < dm_s.number_samples()):
                x, y = dm_s.get_x_y(count_source)
                x_source.append(x)
                y_source.append(y)
                count_source += 1
                count_window += 1
            elif count_target < dm_t.number_samples():
                x, y = dm_t.get_x_y(count_target)
                x_target.append(x)
                y_target.append(y)
                count_target += 1
                count_window += 1
        else:
            batch_counter += 1
            metrics['number_evaluated_samples_source'].append(count_source)
            metrics['number_evaluated_samples_target'].append(count_target)

            # Workaround to avoid empty stream
            if batch_counter > 1:
                if (count_source - metrics['number_evaluated_samples_source'][-2] == 0):
                    x, y = dm_s.get_x_y(np.random.randint(0, count_source if count_source > 1 else 2))
                    x_source.append(x)
                    y_source.append(y)
                if (count_target - metrics['number_evaluated_samples_target'][-2] == 0):
                    x, y = dm_t.get_x_y(np.random.randint(0, count_target if count_target > 1 else 2))
                    x_target.append(x)
                    y_target.append(y)
            # Workaround to avoid empty stream

            x_source = torch.tensor(x_source, dtype=torch.float, device=MyDevice().get())
            y_source = torch.tensor(y_source, dtype=torch.float, device=MyDevice().get())
            x_target = torch.tensor(x_target, dtype=torch.float, device=MyDevice().get())
            y_target = torch.tensor(y_target, dtype=torch.float, device=MyDevice().get())

            # TEST
            if batch_counter > 1:
                metrics['test_time'].append(time.time())
                __test(network=nn, x=x_source, y=y_source,
                       is_source=True, is_discriminative=True, metrics=metrics)
                __test(network=nn, x=x_target, y=y_target,
                       is_source=False, is_discriminative=True, metrics=metrics)
                __test(network=dae, x=x_source,
                       is_source=True, is_discriminative=False, metrics=metrics)
                __test(network=dae, x=x_target,
                       is_source=False, is_discriminative=False, metrics=metrics)

                da.test(x=torch.cat([x_source, x_target]),
                        y=torch.cat([SOURCE_DOMAIN_LABEL.repeat(x_source.shape[0], 1),
                                     TARGET_DOMAIN_LABEL.repeat(x_target.shape[0], 1)]))
                metrics['domain_regression_loss'].append(float(da.loss_value))
                metrics['classification_rate_domain'].append(da.classification_rate)
                metrics['domain_classification_misclassified'].append(da.misclassified)
                metrics['test_time'][-1] = time.time() - metrics['test_time'][-1]

            # TRAIN
            metrics['train_time'].append(time.time())

            common_source, x_target = __force_same_size(torch.cat((x_source.T, y_source.T)).T, x_target, shuffle=False)
            x_source, y_source = common_source.T.split(x_source.shape[1])
            x_source, y_source = x_source.T, y_source.T

            epoch = 1
            while epoch <= internal_epochs:
                xs = x_source.view(-1, x_source.shape[1])
                xt = x_target.view(-1, x_target.shape[1])
                ys = y_source.view(-1, y_source.shape[1])
                # for xs, xt, ys in [(xs.view(1, x_source.shape[1]), x_target.view(1, xt.shape[1]), y_source.view(1, ys.shape[1]))
                #                    for xs, xt, ys in zip(x_source, x_target, y_source)]:
                # Evolving
                if epoch == 1:
                    # Evolving Feature Extraction
                    for i in range(0, 2):
                        if i == 0:
                            __width_evolution(network=dae, x=xs, y=xt, is_gpu=True)
                        elif i == 1:
                            __width_evolution(network=dae, x=xt, y=xs, is_gpu=True)
                        if __grow_nodes(dae, da, nn):
                            __copy_weights(source=dae, targets=[da, nn], layer_numbers=[1], copy_moment=False)
                        elif __prune_nodes(dae, da, nn):
                            __copy_weights(source=dae, targets=[da, nn], layer_numbers=[1], copy_moment=False)

                    # Evolving Source
                    __width_evolution(network=nn, x=xs, y=ys, is_gpu=True)
                    __width_evolution(network=da, x=xs, y=torch.cat(xs.shape[0]*[SOURCE_DOMAIN_LABEL]), is_gpu=True)
                    if not __grow_nodes(da, nn):
                        if __prune_nodes(da):
                            __prune_nodes(nn)
                        elif not __grow_nodes(nn):
                            __prune_nodes(nn)

                    # Evolving Target
                    __width_evolution(network=da, x=xt, y=torch.cat(xt.shape[0]*[TARGET_DOMAIN_LABEL]), is_gpu=True)
                    if not __grow_nodes(da, nn):
                        __prune_nodes(da)

                

                # Denoising AutoEncoder
                __generative(network=dae, x=xs, y=xt)
                __copy_weights(source=dae, targets=[da, nn], layer_numbers=[1], copy_moment=False)

                __generative(network=dae, x=xt, y=xs)
                __copy_weights(source=dae, targets=[da, nn], layer_numbers=[1], copy_moment=False)

                # Domain Discriminator
                da.feedforward(x=xs, y=torch.cat(xs.shape[0]*[SOURCE_DOMAIN_LABEL]), train=True).backpropagate()
                dae.weight[0] = dae.weight[0] - da.learning_rate * da.weight[0].grad.neg()
                dae.bias[0] = dae.bias[0] - da.learning_rate * da.bias[0].grad.neg()
                for weight_no in range(da.number_hidden_layers, 0, -1):
                    da.update_weight(weight_no=weight_no)

                da.feedforward(x=xt, y=torch.cat(xt.shape[0]*[TARGET_DOMAIN_LABEL]), train=True).backpropagate()
                dae.weight[0] = dae.weight[0] - da.learning_rate * da.weight[0].grad.neg()
                dae.bias[0] = dae.bias[0] - da.learning_rate * da.bias[0].grad.neg()
                for weight_no in range(da.number_hidden_layers, 0, -1):
                    da.update_weight(weight_no=weight_no)
                __copy_weights(source=dae, targets=[da, nn], layer_numbers=[1], copy_moment=False)

                # Discriminator
                __discriminative(network=nn, x=xs, y=ys)
                __copy_weights(source=nn, targets=[da, dae], layer_numbers=[1], copy_moment=True)


                epoch += 1
                da.test(x=torch.cat([x_source, x_target]),
                        y=torch.cat([SOURCE_DOMAIN_LABEL.repeat(x_source.shape[0], 1),
                                        TARGET_DOMAIN_LABEL.repeat(x_target.shape[0], 1)]))

            # Metrics
            metrics['train_time'][-1] = time.time() - metrics['train_time'][-1]
            metrics['node_evolution_discriminator'].append(nn.layers[-2])
            metrics['node_evolution_domain_classifier'].append(da.layers[-2])
            metrics['node_evolution_feature_extraction'].append(dae.layers[-2])
            if random.randint(0, 100) > 80:
                print_metrics(batch_counter, metrics, dm_s, dm_t, nn, dae, da)

            # Reset variables for the next batch
            x_source = []
            y_source = []
            x_target = []
            y_target = []
            count_window = 0

    result_string = '%f (T) | %f (S) \t ' \
                    '%f (T) | %f (S) \t ' \
                    '%f (T) | %f (S) \t ' \
                    '%f (T) | %f (S) \t ' \
                    '%f | %d \t ' \
                    '%f | %d \t ' \
                    '%f | %d \t ' \
                    '%f | %f' % (
                        np.mean(metrics['classification_rate_target']),
                        np.mean(metrics['classification_rate_source']),

                        metrics['f1_score_target_micro'][-1],
                        metrics['f1_score_source_micro'][-1],

                        metrics['f1_score_target_macro'][-1],
                        metrics['f1_score_source_macro'][-1],

                        metrics['f1_score_target_weighted'][-1],
                        metrics['f1_score_source_weighted'][-1],

                        np.mean(metrics['node_evolution_feature_extraction']),
                        metrics['node_evolution_feature_extraction'][-1],

                        np.mean(metrics['node_evolution_discriminator']),
                        metrics['node_evolution_discriminator'][-1],

                        np.mean(metrics['node_evolution_domain_classifier']),
                        metrics['node_evolution_domain_classifier'][-1],

                        np.mean(metrics['train_time']),
                        np.sum(metrics['train_time']))

    print('CR Rate (Target) | CR Rate (Source) | \t ' \
          'F1 Score Micro (Target) | F1 Score Micro (Source) | \t ' \
          'F1 Score Macro (Target) | F1 Score Macro (Source) | \t ' \
          'F1 Score Weighted (Target) | F1 Score Weighted (Source) | \t ' \
          'Feature Extractor Node Evolution (mean | final) \t ' \
          'Discriminator Node Evolution (mean | final) \t ' \
          'Domain Classifier Node Evolution (mean | final) \t ' \
          'Train Time (mean | total)')
    print(result_string)

    result = {}
    result['string'] = result_string
    result['classification_rate_source_batch'] = np.nanmean(metrics['classification_rate_source'])
    result['classification_rate_target_batch'] = np.nanmean(metrics['classification_rate_target'])
    result['classification_rate_domain_batch'] = np.nanmean(metrics['classification_rate_domain'])
    result['classification_rate_source_total'] = 1 - np.sum(
        metrics['classification_source_misclassified']) / dm_s.number_samples()
    result['classification_rate_target_total'] = 1 - np.sum(
        metrics['classification_target_misclassified']) / dm_t.number_samples()
    result['classification_rate_domain_total'] = 1 - np.sum(metrics['domain_classification_misclassified']) / (
            dm_s.number_samples() + dm_t.number_samples())
    result['f1_score_target_micro'] = metrics['f1_score_target_micro'][-1]
    result['f1_score_source_micro'] = metrics['f1_score_source_micro'][-1]
    result['f1_score_target_macro'] = metrics['f1_score_target_macro'][-1]
    result['f1_score_source_macro'] = metrics['f1_score_source_macro'][-1]
    result['f1_score_target_weighted'] = metrics['f1_score_target_weighted'][-1]
    result['f1_score_source_weighted'] = metrics['f1_score_source_weighted'][-1]
    result['source_node_mean'] = np.nanmean(metrics['node_evolution_discriminator'])
    result['target_node_mean'] = np.nanmean(metrics['node_evolution_feature_extraction'])
    result['domain_node_mean'] = np.nanmean(metrics['node_evolution_domain_classifier'])
    result['source_node_final'] = metrics['node_evolution_discriminator'][-1]
    result['target_node_final'] = metrics['node_evolution_feature_extraction'][-1]
    result['domain_node_final'] = metrics['node_evolution_domain_classifier'][-1]
    result['train_time_mean'] = np.nanmean(metrics['train_time'])
    result['train_time_final'] = np.nansum(metrics['train_time'])
    result['test_time_mean'] = np.nanmean(metrics['test_time'])
    result['test_time_final'] = np.nansum(metrics['test_time'])
    result['classification_source_loss_mean'] = np.nanmean(metrics['classification_source_loss'])
    result['classification_target_loss_mean'] = np.nanmean(metrics['classification_target_loss'])
    result['reconstruction_source_loss_mean'] = np.nanmean(metrics['reconstruction_source_loss'])
    result['reconstruction_target_loss_mean'] = np.nanmean(metrics['reconstruction_target_loss'])
    result['domain_adaptation_loss_mean'] = np.nanmean(metrics['domain_regression_loss'])

    print()
    print(result)

    __plot_time(metrics['train_time'],
                metrics['test_time'],
                annotation=False)
    __plot_classification_rates(metrics['classification_rate_source'],
                                metrics['classification_rate_target'],
                                metrics['classification_rate_domain'],
                                1 - np.sum(metrics['classification_source_misclassified']) / dm_s.number_samples(),
                                1 - np.sum(metrics['classification_target_misclassified']) / dm_t.number_samples(),
                                1 - np.sum(metrics['domain_classification_misclassified']) / (
                                        dm_s.number_samples() + dm_t.number_samples()),
                                class_number=dm_s.number_classes(),
                                annotation=False)
    __plot_node_evolution(metrics['node_evolution_discriminator'],
                          metrics['node_evolution_domain_classifier'],
                          metrics['node_evolution_feature_extraction'],
                          annotation=False)
    __plot_losses(metrics['classification_source_loss'],
                  metrics['classification_target_loss'],
                  metrics['reconstruction_source_loss'],
                  metrics['reconstruction_target_loss'],
                  metrics['domain_regression_loss'],
                  annotation=False)
    __plot_discriminative_network_significance(nn.BIAS, nn.VAR, annotation=False)
    __plot_domain_classifier_network_significance(da.BIAS, da.VAR, annotation=False)
    __plot_feature_extractor_network_significance(dae.BIAS, dae.VAR, annotation=False)

    return result


def generate_csv_from_dataset(dataset_name: str,
                              n_concept_drift: int = 1,
                              is_source: bool = True,
                              is_one_hot_encoding: bool = True,
                              label_starts_at: int = 0):
    import csv, os
    from tqdm import tqdm
    filename = 'source.csv' if is_source else 'target.csv'

    dm, _ = __load_source_target(source=dataset_name,
                                 target='',
                                 n_source_concept_drift=n_concept_drift)

    try:
        os.remove(filename)
    except:
        pass
    f = open(filename, 'x')
    f.close()

    print('Exporting dataset "%s" as file "%s"' % (dataset_name, filename))
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        pbar = tqdm(total=dm.number_samples())
        for i in range(dm.number_samples()):
            x, y = dm.get_x_y(i)
            temp_y = np.zeros(dm.number_classes() + label_starts_at)
            temp_y[y.argmax() + label_starts_at] = 1
            y = temp_y
            if not is_one_hot_encoding:
                y = np.asarray([y.argmax()])

            writer.writerow(np.concatenate((x, y)).tolist())
            pbar.update(1)
        pbar.close()
    print('Done!')


def generate_arff_from_dataset(source_dataset_name: str,
                               target_dataset_name: str,
                               n_source_concept_drifts: int = 1,
                               n_target_concept_drifts: int = 1,
                               output_filename : str = None):
    import os
    import arff
    if output_filename is not None:
        filename = output_filename
    else:
        filename = 'source_target_melanie.arff'

    dm_s, dm_t = __load_source_target(source=source_dataset_name,
                                      target=target_dataset_name,
                                      n_source_concept_drift=n_source_concept_drifts,
                                      n_target_concept_drift=n_target_concept_drifts)

    try:
        os.remove(filename)
    except:
        pass

    print(f'Exporting datasets {source_dataset_name} and {target_dataset_name} as file {filename}')
    data = []

    count_source = 0
    count_target = 0

    while count_source < dm_s.number_samples() or count_target < dm_t.number_samples():
        source_prob = (dm_s.number_samples() - count_source) / (
            dm_s.number_samples() - count_source + dm_t.number_samples() - count_target + 0.)

        sample = []
        if (np.random.rand() <= source_prob and count_source < dm_s.number_samples()) or (
            count_target >= dm_t.number_samples() and count_source < dm_s.number_samples()):
            x, y = dm_s.get_x_y(count_source)
            count_source += 1
            sample.append(1)
        elif count_target < dm_t.number_samples():
            x, y = dm_t.get_x_y(count_target)
            count_target += 1
            sample.append(0)

        for i in x.tolist():
            sample.append(i)
        sample.append(y.argmax())
        data.append(sample)

    # data = {'data': data,
    #         'relation': f'{source_dataset_name}_{target_dataset_name}',
    #         'attributes': 'something'}
    # with open(filename, 'x') as f:
    #     arff.dump(data, f)
    arff.dump(filename, data, relation=f'{source_dataset_name}_{target_dataset_name}')
    print('done')



def pre_download_benchmarks():
    def print_info(dm):
        print('Number of samples: %d' % dm.number_samples())
        print('Number of features: %d' % dm.number_features())
        print('Number of classes: %d' % dm.number_classes())
        return DataManipulator()

    dm = DataManipulator()
    dm.load_mnist()
    dm = print_info(dm)
    dm.load_usps()
    dm = print_info(dm)
    dm.load_cifar10()
    dm = print_info(dm)
    dm.load_stl10()
    dm = print_info(dm)
    dm.load_london_bike_sharing()
    dm = print_info(dm)
    dm.load_washington_bike_sharing()
    dm = print_info(dm)
    # dm.load_news_popularity_obama_all()
    # dm = print_info(dm)
    # dm.load_news_popularity_economy_all()
    # dm = print_info(dm)
    # dm.load_news_popularity_microsoft_all()
    # dm = print_info(dm)
    # dm.load_news_popularity_palestine_all()
    # dm = print_info(dm)
    # dm.load_amazon_review_fashion()
    # dm = print_info(dm)
    dm.load_amazon_review_all_beauty()
    dm = print_info(dm)
    # dm.load_amazon_review_appliances()
    # dm = print_info(dm)
    # dm.load_amazon_review_arts_crafts_sewing()
    # dm = print_info(dm)
    # dm.load_amazon_review_automotive()
    # dm = print_info(dm)
    # dm.load_amazon_review_cds_vinyl()
    # dm = print_info(dm)
    # dm.load_amazon_review_cellphones_accessories()
    # dm = print_info(dm)
    # dm.load_amazon_review_clothing_shoes_jewelry()
    # dm = print_info(dm)
    # dm.load_amazon_review_digital_music()
    # dm = print_info(dm)
    # dm.load_amazon_review_electronics()
    # dm = print_info(dm)
    # dm.load_amazon_review_gift_card()
    # dm = print_info(dm)
    # dm.load_amazon_review_grocery_gourmet_food()
    # dm = print_info(dm)
    # dm.load_amazon_review_home_kitchen()
    # dm = print_info(dm)
    dm.load_amazon_review_industrial_scientific()
    dm = print_info(dm)
    # dm.load_amazon_review_kindle_store()
    # dm = print_info(dm)
    dm.load_amazon_review_luxury_beauty()
    dm = print_info(dm)
    dm.load_amazon_review_magazine_subscription()
    dm = print_info(dm)
    # dm.load_amazon_review_movies_tv()
    # dm = print_info(dm)
    # dm.load_amazon_review_musical_instruments()
    # dm = print_info(dm)
    # dm.load_amazon_review_office_products()
    # dm = print_info(dm)
    # dm.load_amazon_review_patio_lawn_garden()
    # dm = print_info(dm)
    # dm.load_amazon_review_pet_supplies()
    # dm = print_info(dm)
    # dm.load_amazon_review_prime_pantry()
    # dm = print_info(dm)
    # dm.load_amazon_review_software()
    # dm = print_info(dm)
    # dm.load_amazon_review_sports_outdoors()
    # dm = print_info(dm)
    # dm.load_amazon_review_tools_home_improvements()
    # dm = print_info(dm)
    # dm.load_amazon_review_toys_games()
    # dm = print_info(dm)
    # dm.load_amazon_review_video_games()
    # dm = print_info(dm)
    dm.load_amazon_review_books()
    print_info(dm)


print('ACDC: Autonomous Cross Domain Conversion')
print('')
print('Available methods:')
print('************************************************************')
print('def acdc(%s,%s,%s,%s,%s,%s\n\t)' % (
    '\n\tsource: str',
    '\n\ttarget: str',
    '\n\tn_source_concept_drift: int = 5',
    '\n\tn_target_concept_drift: int = 7',
    '\n\tinternal_epochs: int = 1',
    '\n\tis_gpu: bool = False'))
print(' ')
print('source: String representing the source benchmark')
print('target: String representing the target benchmark')
print('n_source_concept_drift: Number of concept drifts at the source stream')
print('n_target_concept_drift: Number of concept drifts at the target stream')
print('internal_epochs: Number of internal epochs per minibatch')
print(
    'is_gpu: False to run on CPU. True to run on GPU. The paper were generated on CPU. The code is not optimized for GPU. Only runs if you have a huge ammount of GRAM')
print(' ')
print('Returns a dictionary with all results for the run')
print('************************************************************')
print(' ')
print('************************************************************')
print('pre_download_benchmarks()')
print('************************************************************')
print(' ')
print('************************************************************')
print('generate_csv_from_dataset(%s,%s,%s,%s,%s\n\t)' % (
    '\n\tdataset_name: str',
    '\n\tn_concept_drift: int = 1',
    '\n\tis_source: bool = True',
    '\n\tis_one_hot_enconding: bool = True',
    '\n\tlabel_starts_at: int = 0'))
print(' ')
print('dataset_name: String representing which benchmark should be converted to CSV')
print('n_concept_drift: Number of concept drifts applied into the CSV dataset')
print('is_source: True to generate a file "source.csv", False  to generate a file "target.csv"')
print('is_one_hot_enconding: If True, label will be the n last columns in an one-hot-encoding format, if False, label will be the last column as a number')
print('label_starts_at: The smallest label. Usually it is 0, but some source_code, specially made in Matlab, can start from 1')
print('************************************************************')
print(' ')
print('List of possible strings for datasets:')
print(' ')
print('mnist-28: MNIST resized to 28x28, which is original size ~ 784 features')
print('mnist-16: MNIST resized to 16x16 ~ 256 features')
print('usps-28: USPS resized to 28x28 ~ 784 features')
print('usps-16: USPS resized to 16x16, which is original size ~ 256 features')
print('cifar10: CIFAR10 extracted from Resnet ~ 512 features')
print('stl10: STL10 extracted from Resnet ~512 features')
print('amazon-review-all-beauty: Amazon Review | All Beauty | Word2Vec applied ~ 300 features')
print('amazon-review-books: Amazon Review | Books | Word2Vec applied ~ 300 features')
print('amazon-review-industrial-scientific: Amazon Review | Industrial and Scientific | Word2Vec applied ~ 300 features')
print('amazon-review-luxury-beauty: Amazon Review | Luxury Beauty | Word2Vec applied ~ 300 features')
print('amazon-review-magazine-subscription: Amazon Review | Magazine Subscription | Word2Vec applied ~ 300 features')
print('london-bike: London bike sharing dataset ~ 8 features')
print('washington-bike: Washington D.C. bike sharing dataset ~ 8 features')

# pre_download_benchmarks()
# acdc('mnist-28', 'usps-28', 5, 7, 5, False)
# acdc('usps-16', 'mnist-16', 5, 7, 5, False)
# acdc('london-bike', 'washington-bike', 5, 7, 5, False)
# acdc('washington-bike', 'london-bike', 5, 7, 5, False)
# generate_arff_from_dataset('mnist-28', 'usps-28', 5, 7, 'mnist-28_usps-28_1.arff')
# generate_arff_from_dataset('mnist-28', 'usps-28', 5, 7, 'mnist-28_usps-28_2.arff')
# generate_arff_from_dataset('mnist-28', 'usps-28', 5, 7, 'mnist-28_usps-28_3.arff')
# generate_arff_from_dataset('mnist-28', 'usps-28', 5, 7, 'mnist-28_usps-28_4.arff')
# generate_arff_from_dataset('mnist-28', 'usps-28', 5, 7, 'mnist-28_usps-28_5.arff')
#
# generate_arff_from_dataset('usps-16', 'mnist-16', 5, 7, 'usps-16_mnist-16_1.arff')
# generate_arff_from_dataset('usps-16', 'mnist-16', 5, 7, 'usps-16_mnist-16_2.arff')
# generate_arff_from_dataset('usps-16', 'mnist-16', 5, 7, 'usps-16_mnist-16_3.arff')
# generate_arff_from_dataset('usps-16', 'mnist-16', 5, 7, 'usps-16_mnist-16_4.arff')
# generate_arff_from_dataset('usps-16', 'mnist-16', 5, 7, 'usps-16_mnist-16_5.arff')
#
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-books', 5, 7, 'beauty_books_1.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-books', 5, 7, 'beauty_books_2.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-books', 5, 7, 'beauty_books_3.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-books', 5, 7, 'beauty_books_4.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-books', 5, 7, 'beauty_books_5.arff')
#
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-industrial-scientific', 5, 7, 'beauty_industrial_1.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-industrial-scientific', 5, 7, 'beauty_industrial_2.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-industrial-scientific', 5, 7, 'beauty_industrial_3.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-industrial-scientific', 5, 7, 'beauty_industrial_4.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-industrial-scientific', 5, 7, 'beauty_industrial_5.arff')
#
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-luxury-beauty', 5, 7, 'beauty_luxury_1.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-luxury-beauty', 5, 7, 'beauty_luxury_2.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-luxury-beauty', 5, 7, 'beauty_luxury_3.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-luxury-beauty', 5, 7, 'beauty_luxury_4.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-luxury-beauty', 5, 7, 'beauty_luxury_5.arff')
#
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-magazine-subscription', 5, 7, 'beauty_magazine_1.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-magazine-subscription', 5, 7, 'beauty_magazine_2.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-magazine-subscription', 5, 7, 'beauty_magazine_3.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-magazine-subscription', 5, 7, 'beauty_magazine_4.arff')
# generate_arff_from_dataset('amazon-review-all-beauty', 'amazon-review-magazine-subscription', 5, 7, 'beauty_magazine_5.arff')
#
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-all-beauty', 5, 7, 'books_beauty_1.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-all-beauty', 5, 7, 'books_beauty_2.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-all-beauty', 5, 7, 'books_beauty_3.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-all-beauty', 5, 7, 'books_beauty_4.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-all-beauty', 5, 7, 'books_beauty_5.arff')
#
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-industrial-scientific', 5, 7, 'books_industrial_1.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-industrial-scientific', 5, 7, 'books_industrial_2.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-industrial-scientific', 5, 7, 'books_industrial_3.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-industrial-scientific', 5, 7, 'books_industrial_4.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-industrial-scientific', 5, 7, 'books_industrial_5.arff')
#
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-luxury-beauty', 5, 7, 'books_luxury_1.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-luxury-beauty', 5, 7, 'books_luxury_2.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-luxury-beauty', 5, 7, 'books_luxury_3.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-luxury-beauty', 5, 7, 'books_luxury_4.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-luxury-beauty', 5, 7, 'books_luxury_5.arff')
#
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-magazine-subscription', 5, 7, 'books_magazine_1.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-magazine-subscription', 5, 7, 'books_magazine_2.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-magazine-subscription', 5, 7, 'books_magazine_3.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-magazine-subscription', 5, 7, 'books_magazine_4.arff')
# generate_arff_from_dataset('amazon-review-books', 'amazon-review-magazine-subscription', 5, 7, 'books_magazine_5.arff')
#
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-all-beauty', 5, 7, 'industrial_beauty_1.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-all-beauty', 5, 7, 'industrial_beauty_2.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-all-beauty', 5, 7, 'industrial_beauty_3.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-all-beauty', 5, 7, 'industrial_beauty_4.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-all-beauty', 5, 7, 'industrial_beauty_5.arff')
#
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-books', 5, 7, 'industrial_books_1.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-books', 5, 7, 'industrial_books_2.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-books', 5, 7, 'industrial_books_3.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-books', 5, 7, 'industrial_books_4.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-books', 5, 7, 'industrial_books_5.arff')
#
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-luxury-beauty', 5, 7, 'industrial_luxury_1.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-luxury-beauty', 5, 7, 'industrial_luxury_2.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-luxury-beauty', 5, 7, 'industrial_luxury_3.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-luxury-beauty', 5, 7, 'industrial_luxury_4.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-luxury-beauty', 5, 7, 'industrial_luxury_5.arff')
#
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-magazine-subscription', 5, 7, 'industrial_magazine_1.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-magazine-subscription', 5, 7, 'industrial_magazine_2.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-magazine-subscription', 5, 7, 'industrial_magazine_3.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-magazine-subscription', 5, 7, 'industrial_magazine_4.arff')
# generate_arff_from_dataset('amazon-review-industrial-scientific', 'amazon-review-magazine-subscription', 5, 7, 'industrial_magazine_5.arff')
#
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-all-beauty', 5, 7, 'luxury_beauty_1.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-all-beauty', 5, 7, 'luxury_beauty_2.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-all-beauty', 5, 7, 'luxury_beauty_3.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-all-beauty', 5, 7, 'luxury_beauty_4.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-all-beauty', 5, 7, 'luxury_beauty_5.arff')
#
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-books', 5, 7, 'luxury_books_1.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-books', 5, 7, 'luxury_books_2.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-books', 5, 7, 'luxury_books_3.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-books', 5, 7, 'luxury_books_4.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-books', 5, 7, 'luxury_books_5.arff')
#
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-industrial-scientific', 5, 7, 'luxury_industrial_1.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-industrial-scientific', 5, 7, 'luxury_industrial_2.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-industrial-scientific', 5, 7, 'luxury_industrial_3.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-industrial-scientific', 5, 7, 'luxury_industrial_4.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-industrial-scientific', 5, 7, 'luxury_industrial_5.arff')
#
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-magazine-subscription', 5, 7, 'luxury_magazine_1.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-magazine-subscription', 5, 7, 'luxury_magazine_2.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-magazine-subscription', 5, 7, 'luxury_magazine_3.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-magazine-subscription', 5, 7, 'luxury_magazine_4.arff')
# generate_arff_from_dataset('amazon-review-luxury-beauty', 'amazon-review-magazine-subscription', 5, 7, 'luxury_magazine_5.arff')
#
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-all-beauty', 5, 7, 'magazine_beauty_1.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-all-beauty', 5, 7, 'magazine_beauty_2.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-all-beauty', 5, 7, 'magazine_beauty_3.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-all-beauty', 5, 7, 'magazine_beauty_4.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-all-beauty', 5, 7, 'magazine_beauty_5.arff')
#
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-books', 5, 7, 'magazine_books_1.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-books', 5, 7, 'magazine_books_2.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-books', 5, 7, 'magazine_books_3.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-books', 5, 7, 'magazine_books_4.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-books', 5, 7, 'magazine_books_5.arff')
#
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-industrial-scientific', 5, 7, 'magazine_industrial_1.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-industrial-scientific', 5, 7, 'magazine_industrial_2.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-industrial-scientific', 5, 7, 'magazine_industrial_3.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-industrial-scientific', 5, 7, 'magazine_industrial_4.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-industrial-scientific', 5, 7, 'magazine_industrial_5.arff')
#
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-luxury-beauty', 5, 7, 'magazine_luxury_1.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-luxury-beauty', 5, 7, 'magazine_luxury_2.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-luxury-beauty', 5, 7, 'magazine_luxury_3.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-luxury-beauty', 5, 7, 'magazine_luxury_4.arff')
# generate_arff_from_dataset('amazon-review-magazine-subscription', 'amazon-review-luxury-beauty', 5, 7, 'magazine_luxury_5.arff')
#
# generate_arff_from_dataset('cifar10', 'stl10', 5, 7, 'cifar_stl_1.arff')
# generate_arff_from_dataset('cifar10', 'stl10', 5, 7, 'cifar_stl_2.arff')
# generate_arff_from_dataset('cifar10', 'stl10', 5, 7, 'cifar_stl_3.arff')
# generate_arff_from_dataset('cifar10', 'stl10', 5, 7, 'cifar_stl_4.arff')
# generate_arff_from_dataset('cifar10', 'stl10', 5, 7, 'cifar_stl_5.arff')
#
# generate_arff_from_dataset('stl10', 'cifar10', 5, 7, 'stl_cifar_1.arff')
# generate_arff_from_dataset('stl10', 'cifar10', 5, 7, 'stl_cifar_2.arff')
# generate_arff_from_dataset('stl10', 'cifar10', 5, 7, 'stl_cifar_3.arff')
# generate_arff_from_dataset('stl10', 'cifar10', 5, 7, 'stl_cifar_4.arff')
# generate_arff_from_dataset('stl10', 'cifar10', 5, 7, 'stl_cifar_5.arff')
#
# generate_arff_from_dataset('london-bike', 'washington-bike', 5, 7, 'london_washington_1.arff')
# generate_arff_from_dataset('london-bike', 'washington-bike', 5, 7, 'london_washington_2.arff')
# generate_arff_from_dataset('london-bike', 'washington-bike', 5, 7, 'london_washington_3.arff')
# generate_arff_from_dataset('london-bike', 'washington-bike', 5, 7, 'london_washington_4.arff')
# generate_arff_from_dataset('london-bike', 'washington-bike', 5, 7, 'london_washington_5.arff')
#
# generate_arff_from_dataset('washington-bike', 'london-bike', 5, 7, 'washington_london_1.arff')
# generate_arff_from_dataset('washington-bike', 'london-bike', 5, 7, 'washington_london_2.arff')
# generate_arff_from_dataset('washington-bike', 'london-bike', 5, 7, 'washington_london_3.arff')
# generate_arff_from_dataset('washington-bike', 'london-bike', 5, 7, 'washington_london_4.arff')
# generate_arff_from_dataset('washington-bike', 'london-bike', 5, 7, 'washington_london_5.arff')