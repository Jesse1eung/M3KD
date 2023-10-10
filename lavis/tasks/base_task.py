"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import random
import torch


import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample

torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        # loss = model(samples)["loss"]
        output = model(samples)
        return output

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, **kwargs):
        pass

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            eval_output = self.valid_step(model=model, samples=samples)
            results.extend(eval_output)

        # dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        accelerator=None,
    ):
        # length = len(data_loader["txt_train"])

        if 'txt_train' in data_loader and 'img_train' in data_loader:
            length = accum_grad_iters * len(data_loader["train"])
            # length = accum_grad_iters * len(data_loader["img_train"])
        elif 'txt_train' in data_loader:
            if 'train' not in data_loader.keys():
                # length = len(data_loader["txt_train"])
                length = 5000
            else:
                length = 2 * len(data_loader["train"]) if accum_grad_iters ==2 else len(data_loader["train"])
        elif 'img_train' in data_loader:
            if 'train' not in data_loader.keys():
                length = len(data_loader["img_train"])
                print(length,'dataimg')
                # length = 5000
            else:
                length = 2 * len(data_loader["train"]) if accum_grad_iters ==2 else len(data_loader["train"])
        else:
            length = len(data_loader["train"])
            accum_grad_iters = 1
        # length = len(data_loader["train"]) + len(data_loader["txt_train"])
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=length,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            accelerator=accelerator,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        accelerator=None
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        vl_data_loader = data_loader["train"]
        # vl_data_loader = data_loader["txt_train"]
        # vl_data_loader = data_loader["img_train"]
        if 'txt_train' in data_loader.keys():
            l_data_loader = data_loader["txt_train"]
            if 'train' in data_loader.keys():

                vl_data_loader = data_loader["train"]
            else:

                vl_data_loader = data_loader["txt_train"]
        if 'img_train' in data_loader.keys():
            i_data_loader = data_loader["img_train"]
            if 'train' in data_loader.keys():
                print('trainvl')
                vl_data_loader = data_loader["train"]
            else:
                # print('trainimg')
                vl_data_loader = data_loader["img_train"]

        if not hasattr(vl_data_loader, "__next__"):
            # convert to iterator if not already
            print('not has next attr ')
            if 'train' in data_loader.keys():
                vl_data_loader = iter(vl_data_loader)
            if 'txt_train' in data_loader.keys():
                l_data_loader = iter(l_data_loader)
            if 'img_train' in data_loader.keys():
                i_data_loader = iter(i_data_loader)
        sw = 0


        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("img", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("itc", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("itm", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("logits", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("reps", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("itc_distill", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        metric_logger.add_meter("it_hid", SmoothedValue(window_size=1, fmt="{value:.4f}"))
        # metric_logger.add_meter("sims_i2t_T", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            if i==0:
                if 'txt_train' in data_loader.keys() and epoch==40:
                    for j in range(len(data_loader["train"]) * epoch):
                        sam = next(l_data_loader)
                    print(' jump to step of txttrain:',len(data_loader["train"]) * epoch)
            if sw == 0 :
                if 'train' in data_loader.keys():
                    dl =vl_data_loader
                    sw=1
                elif 'txt_train' in data_loader.keys():
                    sw = 1
                    dl = l_data_loader
                else:
                    sw = 2
                    dl = i_data_loader
                if 'txt_train' not in data_loader.keys() and 'img_train' not in data_loader.keys():
                    sw = 0
            elif sw == 1 and 'txt_train' in data_loader.keys():
                dl = l_data_loader
                sw = 2 if 'img_train' in data_loader.keys() else 0 if 'train' in data_loader.keys() else 1
            elif sw == 2:
                dl = i_data_loader
                sw = 0 if 'train' in data_loader.keys() else 1 if 'txt_train' in data_loader.keys() else 2
            # if sw == 0:
            #     dl = vl_data_loader
            #     sw=2
            # else:
            #     dl = i_data_loader
            #     sw=0


            # try:
            #     samples = next(dl)
            # except StopIteration:

            samples = next(dl)
            # continue

            # samples = samples.to(model.device)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):

                output = self.train_step(model=model, samples=samples)
                loss = output["loss"]
                # if torch.isnan(loss):
                #     print('epoch ',epoch, 'step ',i, 'nan.')
                #     continue
                if "loss_img" in output:
                    loss_img = output["loss_img"]
                else:
                    loss_img = torch.tensor([0.])

                if "loss_itc" in output:
                    loss_itc = output["loss_itc"]
                else:
                    loss_itc = torch.tensor([0.])

                if "loss_itm" in output:
                    loss_itm = output["loss_itm"]
                else:
                    loss_itm = torch.tensor([0.])

                if "loss_logits" in output:
                    loss_logits = output["loss_logits"]
                else:
                    loss_logits = torch.tensor([0.])
                if "loss_reps" in output:
                    loss_reps = output["loss_reps"]
                else:
                    loss_reps = torch.tensor([0.])
                if "loss_itc_distill" in output:
                    loss_itc_distill = output["loss_itc_distill"]
                    loss_it_hid = output["loss_it_hid"]
                else:
                    loss_itc_distill = torch.tensor([0.])
                    loss_it_hid = torch.tensor([0.])

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                # loss = loss * 2 / accum_grad_iters
                loss.backward()
                # accelerator.backward(loss)
            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                        # for group in optimizer.param_groups:
                        #     for param in group["params"]:
                            if param.grad is None:
                                continue
                            if param.grad.is_sparse:
                                if param.grad.dtype is torch.float16:
                                    param.grad = param.grad.coalesce()
                                to_unscale = param.grad._values()
                            else:
                                to_unscale = param.grad
                            v = to_unscale.clone().abs().max()
                            if torch.isinf(v) or torch.isnan(v):
                                print('INF grad in', name,v,loss.item(), 'of step', i, '!!!')
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # clip_norm = 3.0
                    # if clip_norm > 0.0:
                    #     if accelerator.sync_gradients:
                    #         # print('sysn grad true')
                    #         accelerator.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            # for group in optimizer.param_groups:
                            #     for param in group["params"]:
                            if param.grad is None:
                                continue
                            if param.grad.is_sparse:
                                if param.grad.dtype is torch.float16:
                                    param.grad = param.grad.coalesce()
                                to_unscale = param.grad._values()
                            else:
                                to_unscale = param.grad
                            v = to_unscale.clone().abs().max()
                            if torch.isinf(v) or torch.isnan(v):
                                print('INF grad in', name,v,loss.item(), 'of step', i, '!!!')
                    optimizer.step()
                # print(loss.isnan().any())
                optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(img=loss_img.item() if type(loss_img)!=float else loss_img)
            metric_logger.update(itc=loss_itc.item() if type(loss_itc)!=float else loss_itc)
            metric_logger.update(itm=loss_itm.item() if type(loss_itm)!=float else loss_itm)
            metric_logger.update(logits=loss_logits.item() if type(loss_logits)!=float else loss_logits)
            metric_logger.update(reps=loss_reps.item()if type(loss_reps)!=float else loss_reps)
            metric_logger.update(itc_distill=loss_itc_distill.item()if type(loss_itc_distill)!=float else loss_itc_distill)
            metric_logger.update(it_hid=loss_it_hid.item()if type(loss_it_hid)!=float else loss_it_hid)
            # metric_logger.update(sims_i2t_T=output['sims']['sim_i2t'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.5f}".format(meter.global_avg) if k=='lr' else "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file

    # def clip_norm_step(self, params):
    #     if

