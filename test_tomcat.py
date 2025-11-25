import json
import os
import sys
from itertools import product

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import hmean
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import cv2
import wandb

import torch.nn.functional as F
import operator
import torch.nn as nn

from utils import *
from parameters import parser
from dataset import CompositionDataset
from model.model_factory import get_model


class TextKAM(nn.Module):
    def __init__(self, text_feats, device):
        super(TextKAM, self).__init__()
        self.dim1, self.dim2 = text_feats.shape
        self.residual = nn.Parameter(torch.zeros([self.dim1, self.dim2], dtype=text_feats.dtype).to(device), requires_grad=True)

    def forward(self, x, weight):
        weight_reshaped = weight.view(-1, 1)
        x_new = x.clone() + weight_reshaped * self.residual
        x_new = F.normalize(x_new, dim=-1)
        return x_new

class VisualKAM(nn.Module):
    def __init__(self, text_feats, device):
        super(VisualKAM, self).__init__()
        self.dim1, self.dim2 = text_feats.shape
        self.residual = nn.Parameter(torch.zeros([self.dim1, self.dim2], dtype=text_feats.dtype).to(device), requires_grad=True)

    def forward(self, x, weight, all_classes):
        weight_reshaped = weight.view(-1, 1)
        x_new = x.clone() + weight_reshaped * self.residual[all_classes]
        x_new = F.normalize(x_new, dim=-1)
        return x_new

def contrastive_loss( x: torch.Tensor, y: torch.Tensor, temperature):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    batch_size = x.shape[0]

    similarity_matrix = torch.mm(x, y.t()) * temperature.to(dtype=x.dtype)  # (batch_size, batch_size)

    labels = torch.arange(batch_size, device=x.device)

    loss_x = F.cross_entropy(similarity_matrix, labels)
    loss_y = F.cross_entropy(similarity_matrix.t(), labels)

    return (loss_x + loss_y) / 2

def self_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_clip_logits(img_feats, model, text_feats):
    clip_logits = model.cos_sim_func_4com_cls(img_feats, text_feats)

    entropy = self_entropy(clip_logits)
    pred = clip_logits.argmax(dim=1)[0].item()  # int(clip_logits.topk(1, 1, True, True)[1].t()[0])

    return clip_logits, entropy, pred

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return

def cache_key_value(image_features, cache, clip_weights, device):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            # Compute the prototype of the class
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_values.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0)
        cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(0)))
        cache_values = cache_values.to(cache_keys.dtype)
        cache_values = cache_values.to(device)
        return cache_keys, cache_values, all_classes

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, device):
    affinity = image_features @ cache_keys.T
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits.to(device)




def adaptive_update_weight(img_feats, text_feats, alpha=10):
    similarity = img_feats @ text_feats.T
    x = 1 / (1 + torch.exp(alpha * similarity))  # sigmoid(-beta * s)
    return x


def predict_logits_text_first_with_tomcat(model, dataset, config):
    """Function to predict the cosine similarities between the
    images and the attribute-object representations. The function
    also returns the ground truth for attributes, objects, and pair
    of attribute-objects.

    Args:
        model (nn.Module): the model
        text_rep (nn.Tensor): the attribute-object representations.
        dataset (CompositionDataset): the composition dataset (validation/test)
        device (str): the device (either cpu/cuda:0)
        config (argparse.ArgumentParser): config/args

    Returns:
        tuple: the logits, attribute labels, object labels,
            pair attribute-object labels
    """
    model.eval()
    device = config.device
    cpu = config.cpu_cache
    use_cache = config.use_img_cache

    all_attr_gt, all_obj_gt, all_pair_gt = (
        [],
        [],
        [],
    )
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    # print(text_rep.shape)
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in pairs_dataset]).to(config.device)
    dataloader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=True,
        num_workers=config.num_workers)
    all_logits = torch.Tensor()
    with torch.no_grad():
        text_feats = []
        num_text_batch = pairs.shape[0] // config.text_encoder_batch_size
        for i_text_batch in range(num_text_batch):
            cur_pair = pairs[i_text_batch * config.text_encoder_batch_size:(
                     i_text_batch + 1) * config.text_encoder_batch_size,
                       :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            text_feats.append(cur_text_feats)
        if pairs.shape[0] % config.text_encoder_batch_size != 0:
            cur_pair = pairs[num_text_batch * config.text_encoder_batch_size:, :]
            cur_text_feats = model.encode_text_for_open(cur_pair)
            text_feats.append(cur_text_feats)

        text_feats = torch.cat(text_feats, dim=0)
        text_feats = F.normalize(text_feats, dim=-1)
    model.release_text_encoder()

    #---------------

    pos_params = {
        'shot_capacity': config.shot_capacity,
        'alpha': config.alpha,
        'beta': config.beta,
    }


    text_kam = TextKAM(text_feats, device)
    optimizer_t = torch.optim.AdamW([
        {'params': text_kam.parameters(), 'lr': config.text_lr, 'eps': config.eps, 'weight_decay': config.wd}
    ])

    if use_cache:
        visual_kam = VisualKAM(text_feats, 'cpu' if cpu else device)
        pos_cache = {}
        optimizer_i = torch.optim.AdamW([
            {'params': visual_kam.parameters(), 'lr': config.image_lr, 'eps': config.eps, 'weight_decay': config.wd}
        ])

    for idx, data in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Testing"
    ):

        data[0] = data[0].to(device)
        with torch.no_grad():
            img_feats, _ = model.encode_image(data[0].type(model.clip.dtype))
            img_feats = F.normalize(img_feats, dim=-1)
            text_weight = adaptive_update_weight(img_feats, text_feats, config.theta)
        new_text_feats = text_kam(text_feats, text_weight)
        clip_logits, entropy, pred = get_clip_logits(img_feats, model, new_text_feats)

        if use_cache:
            a = img_feats.to('cpu')
            b = entropy.to('cpu')
            update_cache(pos_cache, pred,
                         [a if cpu else img_feats, b if cpu else entropy],
                         pos_params['shot_capacity'])
            pos_cache_keys, pos_cache_values, all_classes = cache_key_value(a if cpu else img_feats, pos_cache,
                                                                            new_text_feats, 'cpu' if cpu else device)
            with torch.no_grad():
                cache_weight = adaptive_update_weight(a if cpu else img_feats, pos_cache_keys, config.theta)
            new_pos_cache_keys = visual_kam(pos_cache_keys, cache_weight, all_classes)

            clip_logits = clip_logits + compute_cache_logits(a if cpu else img_feats, new_pos_cache_keys,
                                                             pos_cache_values, pos_params['alpha'], pos_params['beta'], device)
            entropy = self_entropy(clip_logits)

        loss = entropy
        if config.use_align_loss:
            image2text_loss = contrastive_loss(new_pos_cache_keys.to(config.device), new_text_feats[all_classes, :],
                                               model.clip.logit_scale.exp())
            loss = loss + config.align_loss_weight * image2text_loss


        optimizer_t.zero_grad()
        if config.use_img_cache:
            optimizer_i.zero_grad()
        loss.backward() #retain_graph=True)
        optimizer_t.step()
        if config.use_img_cache:
            optimizer_i.step()

        if config.use_wandb:
                wandb.log({'loss': loss.item()})


        attr_truth, obj_truth, pair_truth = data[1], data[2], data[3]
        clip_logits = clip_logits.cpu()

        all_logits = torch.cat([all_logits, clip_logits], dim=0)
        all_attr_gt.append(attr_truth)
        all_obj_gt.append(obj_truth)
        all_pair_gt.append(pair_truth)

    all_attr_gt, all_obj_gt, all_pair_gt = (
        torch.cat(all_attr_gt).to("cpu"),
        torch.cat(all_obj_gt).to("cpu"),
        torch.cat(all_pair_gt).to("cpu"),
    )

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt