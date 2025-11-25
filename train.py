#
import argparse
import csv
import os
import pickle
import pprint
import sys

from torch.cuda.amp import autocast
import numpy as np
import torch
import tqdm
import wandb
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser
from datetime import datetime
from os.path import join as ospj

# from test import *
import test as test
from dataset import CompositionDataset
from utils import *

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset,train_dataloader, scheduler):
    best_val_AUC = 0
    best_test_AUC = 0

    final_model_state = None
    results = []
    train_losses = []

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).to(config.device)

    for epoch in range(0, config.epoch_start):
        scheduler = step_scheduler(scheduler, config, len(train_dataloader)-1, len(train_dataloader))

    for epoch in range(config.epoch_start, config.epochs):
        model.train()
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (epoch)
        )


        epoch_train_losses = []
        for bid, batch in enumerate(train_dataloader):
            batch[0] = batch[0].to(config.device)
            if config.use_mixed_precision:
                with autocast(dtype=torch.bfloat16):
                    loss = model(batch, train_pairs)

            #loss = model.loss_calu(predict, batch)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()

            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss)

            progress_bar.set_postfix({"train loss": torch.stack(epoch_train_losses[-50:]).mean().item()})
            progress_bar.update()
            # break

        each_train_loss = torch.stack(epoch_train_losses).mean()
        progress_bar.close()
        progress_bar.write(f"epoch {epoch} train loss {each_train_loss.item()}")
        train_losses.append(each_train_loss.item())

        # yxd: save ckpt of the newest epoch
        torch.save(model.state_dict(), os.path.join(config.save_path, f"newest_model.pt"))
        config.current_epoch = epoch

        result = {}
        result['train_loss'] = each_train_loss.item()

        print("Epoch " + str(epoch) + " Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        for key, value in val_result.items():
            result['val_' + key] = value

        print("Epoch " + str(epoch) + " Evaluating test dataset:")
        test_result = evaluate(model, test_dataset, config)
        for key, value in test_result.items():
            result['test_' + key] = value

        results.append(result)

        if config.val_metric == 'best_AUC' and val_result['AUC'] > best_val_AUC:
            best_val_AUC = val_result['AUC']
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))

        if config.val_metric == 'best_AUC' and test_result['AUC'] > best_test_AUC:
            best_test_AUC = test_result['AUC']
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "test_best.pt"))

        if epoch == config.epochs - 1:
            final_model_state = model.state_dict()

        if config.use_wandb:
            wandb.log(result)
        print('')

    for epoch in range(len(results)):
        with open(ospj(config.save_path, 'logs.csv'), 'a') as f:
            w = csv.DictWriter(f, results[epoch].keys())
            if epoch == 0:
                w.writeheader()
            w.writerow(results[epoch])
    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None, device = config.device)
    with autocast(dtype=torch.bfloat16):
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test.predict_logits_text_first(
                model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    print(result)
    return test_saved_results



if __name__ == "__main__":
    config = parser.parse_args()
    if config.cfg:
        load_args(config.cfg, config)

    if config.use_wandb:
        os.environ["WANDB_MODE"] = config.wandb_net
        wandb.init(project='Troika-' + config.dataset, config={"start_epoch": config.epoch_start})

    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Beginning Time：" + str(formatted_now))
    print("Programme Path: ", os.path.abspath(sys.argv[0]))
    print("Running Command: ", " ".join(sys.argv))
    for k, v in vars(config).items():
        print(k, ': ', v)

    os.makedirs(config.save_path, exist_ok=True)

    # set the seed value
    set_seed(config.seed)

    dataset_path = config.dataset_path

    train_dataset = CompositionDataset(dataset_path,
                                       phase='train',
                                       split='compositional-split-natural',
                                       same_prim_sample=config.same_prim_sample,
                                       )

    val_dataset = CompositionDataset(dataset_path,
                                     phase='val',
                                     split='compositional-split-natural',
                                     )

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural',
                                      )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    allattrs = train_dataset.attrs
    allobj = train_dataset.objs
    classes = [cla.replace(".", " ").lower() for cla in allobj]
    attributes = [attr.replace(".", " ").lower() for attr in allattrs]
    offset = len(attributes)

    model = get_model(config, attributes=attributes, classes=classes, offset=offset)
    model.to(config.device)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_dataloader))

    if config.epoch_start > 0:
        model.load_state_dict(torch.load(config.load_model))
    try:
        train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset, train_dataloader, scheduler)

    finally:
        write_json(os.path.join(config.save_path, "config.json"), vars(config))

        if config.use_wandb:
            wandb.finish()

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        print("Ending Time：" + str(formatted_now))
        print("Done!")
