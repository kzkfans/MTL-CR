import itertools

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Trainer
import argparse
from typing import Optional
from data.data_collator import collate_fn
from transformers.trainer import *
from weighting.DWA import DWA

class CodeTrainer(Seq2SeqTrainer):

    def __init__(self, main_args: argparse.Namespace, code_vocab, ast_vocab, nl_vocab, task, **kwargs):
        super(CodeTrainer, self).__init__(**kwargs)
        self.main_args = main_args
        self.code_vocab = code_vocab
        self.ast_vocab = ast_vocab
        self.nl_vocab = nl_vocab
        self.task = task

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.main_args.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset:
            self.eval_dataset = eval_dataset
        return DataLoader(dataset=self.eval_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return DataLoader(dataset=test_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def set_task(self, task):
        self.task = task

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.
    #
    #     Works both with or without labels.
    #     """
    #     prediction_loss_only = (
    #         prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
    #     )
    #
    #     # if eval is called w/o train init deepspeed here
    #     if self.args.deepspeed and not self.deepspeed:
    #
    #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
    #         # from the checkpoint eventually
    #         deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine
    #         # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
    #         # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
    #         # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
    #         deepspeed_engine.optimizer.optimizer = None
    #         deepspeed_engine.lr_scheduler = None
    #
    #     model = self._wrap_model(self.model, training=False)
    #
    #     # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
    #     # ``train`` is running, halve it first and then put on device
    #     if not self.is_in_train and self.args.fp16_full_eval:
    #         model = model.half().to(self.args.device)
    #
    #     batch_size = dataloader.batch_size
    #
    #     logger.info(f"***** Running {description} *****")
    #     if isinstance(dataloader.dataset, collections.abc.Sized):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")
    #
    #     model.eval()
    #
    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = dataloader.dataset
    #
    #     if is_torch_tpu_available():
    #         dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)
    #
    #     if self.args.past_index >= 0:
    #         self._past = None
    #
    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     # Will be useful when we have an iterable dataset so don't know its length.
    #
    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #         # 查看每个值的形状
    #         for key, value in inputs.items():
    #             if isinstance(value, torch.Tensor):
    #                 print(f"{key}: {value.shape}")
    #
    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         print('logits', logits.shape, type(logits))
    #         print('labels', labels.shape, type(labels))
    #         # Update containers on host
    #         if loss is not None:
    #             losses = self._nested_gather(loss.repeat(batch_size))
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if logits is not None:
    #             logits = self._pad_across_processes(logits)
    #             logits = self._nested_gather(logits)
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #             labels = self._nested_gather(labels)
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)
    #
    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )
    #
    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, labels_host = None, None, None
    #
    #     if self.args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")
    #
    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #
    #     # Number of samples
    #     if not isinstance(eval_dataset, IterableDataset):
    #         num_samples = len(eval_dataset)
    #     elif isinstance(eval_dataset, IterableDatasetShard):
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         num_samples = observed_num_examples
    #
    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #
    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}
    #
    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)
    #
    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
    #
    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
    #     # preds, label == None
    #
    #     print('all_preds', all_preds.shape)
    #     print('all_labels', all_labels.shape)
    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    #
    # def prediction_step(
    #         self,
    #         model: nn.Module,
    #         inputs: Dict[str, Union[torch.Tensor, Any]],
    #         prediction_loss_only: bool,
    #         ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     """
    #     Perform an evaluation step on :obj:`model` using obj:`inputs`.
    #
    #     Subclass and override to inject custom behavior.
    #
    #     Args:
    #         model (:obj:`nn.Module`):
    #             The model to evaluate.
    #         inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
    #             The inputs and targets of the model.
    #
    #             The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
    #             argument :obj:`labels`. Check your model's documentation for all accepted arguments.
    #         prediction_loss_only (:obj:`bool`):
    #             Whether or not to return the loss only.
    #
    #     Return:
    #         Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
    #         labels (each being optional).
    #     """
    #     print('here!!!!!')
    #     print('predict_with_generate', self.args.predict_with_generate)
    #     print('prediction_loss_only', prediction_loss_only)
    #     if not self.args.predict_with_generate or prediction_loss_only:
    #         return super().prediction_step(
    #             model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
    #         )
    #
    #     has_labels = "labels" in inputs
    #     inputs = self._prepare_inputs(inputs)
    #
    #     # XXX: adapt synced_gpus for fairscale as well
    #     gen_kwargs = {
    #         "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
    #         "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
    #         "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
    #     }
    #     print('gen_kwargs', gen_kwargs)
    #
    #     generated_tokens = self.model.generate(
    #         inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         **gen_kwargs,
    #     )
    #     # in case the batch is shorter than max length, the output should be padded
    #     if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
    #         generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
    #
    #     with torch.no_grad():
    #         if self.use_amp:
    #             print(1)
    #             with autocast():
    #                 outputs = model(**inputs)
    #         else:
    #             print(2)
    #             outputs = model(**inputs)
    #         if has_labels:
    #             if self.label_smoother is not None:
    #                 print(3)
    #                 loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
    #             else:
    #                 print(4)
    #                 loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
    #         else:
    #             print(5)
    #             loss = None
    #
    #     if self.args.prediction_loss_only:
    #         return (loss, None, None)
    #
    #     labels = inputs["labels"]
    #     if labels.shape[-1] < gen_kwargs["max_length"]:
    #         print(6)
    #         labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
    #
    #     return (loss, generated_tokens, labels)

class CodeCLSTrainer(Trainer):

    def __init__(self, main_args: argparse.Namespace, code_vocab, ast_vocab, nl_vocab, task, **kwargs):
        super(CodeCLSTrainer, self).__init__(**kwargs)
        self.main_args = main_args
        self.code_vocab = code_vocab
        self.ast_vocab = ast_vocab
        self.nl_vocab = nl_vocab
        self.task = task

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.main_args.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset:
            self.eval_dataset = eval_dataset
        return DataLoader(dataset=self.eval_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return DataLoader(dataset=test_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args,
                                                              task=self.task,
                                                              code_vocab=self.code_vocab,
                                                              nl_vocab=self.nl_vocab,
                                                              ast_vocab=self.ast_vocab))

    def set_task(self, task):
        self.task = task

def get_weighting_method(method_name, task_num, device=None):
    if method_name == "dwa":
        return DWA(task_num=task_num, device=device)
    else:
        raise ValueError(f"Unknown weighting method: {method_name}")
class CustomTrainer(CodeTrainer):
    def __init__(self,train_datasets, eval_datasets, compute_metrics_dict, weighting_method=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets

        self.compute_metrics_dict = compute_metrics_dict
        self.train_dataset = train_datasets[2]

        self.loss_weights = nn.Parameter(torch.ones(3) / 3)

        self.task_num = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.weighting_method = get_weighting_method(weighting_method, task_num=self.task_num, device=self.device)
        self.model.weighting_method.init_param()
    def get_train_dataloader(self):
        bf = DataLoader(dataset=self.train_datasets[0],
                         batch_size=self.main_args.batch_size,
                         shuffle=True,
                         collate_fn=lambda batch: collate_fn(batch,
                                                             args=self.main_args,
                                                             task='bug_fix',
                                                             code_vocab=self.code_vocab,
                                                             nl_vocab=self.nl_vocab,
                                                             ast_vocab=self.ast_vocab))
        com = DataLoader(dataset=self.train_datasets[1],
                       batch_size=self.main_args.batch_size,
                       shuffle=True,
                       collate_fn=lambda batch: collate_fn(batch,
                                                           args=self.main_args,
                                                           task='completion',
                                                           code_vocab=self.code_vocab,
                                                           nl_vocab=self.nl_vocab,
                                                           ast_vocab=self.ast_vocab))
        tran = DataLoader(dataset=self.train_datasets[2],
                       batch_size=self.main_args.batch_size,
                       shuffle=True,
                       collate_fn=lambda batch: collate_fn(batch,
                                                           args=self.main_args,
                                                           task='translation',
                                                           code_vocab=self.code_vocab,
                                                           nl_vocab=self.nl_vocab,
                                                           ast_vocab=self.ast_vocab))

        zipdataloader = ZipDataLoader([bf, com, tran])

        # 返回一个自定义的数据加载器
        return zipdataloader

    def get_eval_dataloader(self, eval_dataset):
        if eval_dataset:
            self.eval_dataset = eval_dataset
        bf = DataLoader(dataset=self.eval_datasets[0],
                         batch_size=self.main_args.batch_size,
                         shuffle=True,
                         collate_fn=lambda batch: collate_fn(batch,
                                                             args=self.main_args,
                                                             task='bug_fix',
                                                             code_vocab=self.code_vocab,
                                                             nl_vocab=self.nl_vocab,
                                                             ast_vocab=self.ast_vocab))
        com = DataLoader(dataset=self.eval_datasets[1],
                       batch_size=self.main_args.batch_size,
                       shuffle=True,
                       collate_fn=lambda batch: collate_fn(batch,
                                                           args=self.main_args,
                                                           task='completion',
                                                           code_vocab=self.code_vocab,
                                                           nl_vocab=self.nl_vocab,
                                                           ast_vocab=self.ast_vocab))
        tran = DataLoader(dataset=self.eval_datasets[2],
                       batch_size=self.main_args.batch_size,
                       shuffle=True,
                       collate_fn=lambda batch: collate_fn(batch,
                                                           args=self.main_args,
                                                           task='translation',
                                                           code_vocab=self.code_vocab,
                                                           nl_vocab=self.nl_vocab,
                                                           ast_vocab=self.ast_vocab))

        return ZipDataLoader([bf, com, tran])

    def compute_loss(self, model, inputs, return_outputs=False):
        if type(list(inputs.values())[0]) == dict:
            task_outputs = {}
            task_losses = []  # 用于存储每个任务的损失
            for i, task_key in enumerate(["batch1", "batch2", "batch3"]):
                batch = inputs[task_key]
                outputs = model(**batch)  # 前向传播
                


                task_loss = outputs.loss.mean()
                task_losses.append(task_loss)  # 将任务损失添加到列表中
                task_outputs[task_key] = outputs

            task_losses = torch.stack(task_losses).squeeze(0)
            model.module.weighting_method.current_step_losses = (
                task_losses.detach().cpu().tolist()
            )
            loss, weights= self.model.weighting_method.backward(task_losses)

            weights = torch.tensor(weights, device=task_losses.device)

            total_loss = loss
        else:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        if not return_outputs:
            return total_loss
        return total_loss, task_outputs


    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, adapted for multi-task training.

        Args:
            dataloader: DataLoader for evaluation.
            description: Description of the evaluation phase.
            prediction_loss_only: Whether to return only the loss.
            ignore_keys: Keys to ignore in the model output.
            metric_key_prefix: Prefix for metric keys.

        Returns:
            EvalLoopOutput: Evaluation results.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )
        # Initialize deepspeed if needed
        if self.args.deepspeed and not self.deepspeed:
            # print(1)
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # Handle fp16 evaluation
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        losses_host = None
        preds_host = {}
        labels_host = {}
        all_losses = None
        all_preds = {}
        all_labels = {}
        all_task_losses = {}

        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss_raw = []
            logits = {}
            labels = {}
            i = 0
            for task_key, batch in inputs.items():

                i += 1
                tloss, tlogits, tlabels = self.prediction_step(model, batch, prediction_loss_only,
                                                            ignore_keys=ignore_keys)

                loss_raw.append(tloss)

                logits[task_key] = tlogits
                labels[task_key] = tlabels
                all_task_losses[task_key] = tloss


            if loss_raw:

                task_losses = torch.stack(loss_raw)

                loss, weights = self.model.weighting_method.backward(task_losses)


            else:
                loss = None

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)

            if logits is not None:
                for task_key, task_logits in logits.items():
                    task_logits = self._pad_across_processes(task_logits)
                    task_logits = self._nested_gather(task_logits)
                    if task_key not in preds_host:
                        preds_host[task_key] = task_logits
                    else:
                        preds_host[task_key] = nested_concat(preds_host[task_key], task_logits, padding_index=-100)


            if labels is not None:
                for task_key, task_labels in labels.items():
                    task_labels = self._pad_across_processes(task_labels)
                    task_labels = self._nested_gather(task_labels)
                    if task_key not in labels_host:
                        labels_host[task_key] = task_labels
                    else:
                        labels_host[task_key] = nested_concat(labels_host[task_key], task_labels,
                                                              padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)


            if losses_host is not None:
                losses = nested_numpify(losses_host.detach())
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

            if preds_host is not None:
                for task_key, task_preds in preds_host.items():
                    task_preds = nested_numpify(task_preds)
                    # print(task_key, preds_host[task_key])
                    if task_key not in all_preds:
                        all_preds[task_key] = task_preds
                    else:
                        all_preds[task_key] = nested_concat(all_preds[task_key], task_preds, padding_index=-100)


            if labels_host is not None:
                for task_key, task_labels in labels_host.items():
                    task_labels = nested_numpify(task_labels)
                    if task_key not in all_labels:
                        all_labels[task_key] = task_labels
                    else:
                        all_labels[task_key] = nested_concat(all_labels[task_key], task_labels,
                                                             padding_index=-100)

            losses_host, preds_host, labels_host = None, {}, {}


        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        if all_losses is not None:
            all_losses = all_losses[:num_samples]

        if all_preds is not None:
            for task_key, task_preds in all_preds.items():
                all_preds[task_key] = nested_truncate(task_preds, num_samples)

        if all_labels is not None:
            for task_key, task_labels in all_labels.items():
                all_labels[task_key] = nested_truncate(task_labels, num_samples)

        metrics = {}
        if self.compute_metrics_dict is not None and all_preds is not None and all_labels is not None:
            for task_key, task_preds in all_preds.items():

                metrics[f"{metric_key_prefix}_{task_key}_loss"] = all_task_losses[task_key].mean().item()
                task_labels = all_labels.get(task_key)
                if task_labels is not None:
                    compute_metrics_func = self.compute_metrics_dict.get(task_key)
                    if compute_metrics_func is not None:
                        task_metrics = compute_metrics_func(
                            EvalPrediction(predictions=task_preds, label_ids=task_labels))
                        # print(task_metrics)
                        for metric_name, metric_value in task_metrics.items():
                            metrics[f"{metric_key_prefix}_{task_key}_{metric_name}"] = metric_value
            task_scores = [
                (metrics.get("eval_batch1_bleu", 0) + metrics.get("eval_batch1_accuracy", 0)) / 2,
                metrics.get("eval_batch2_accuracy", 0),
                (metrics.get("eval_batch3_bleu", 0) + metrics.get("eval_batch3_accuracy", 0)) / 2,
            ]
            composite_score = sum(task_scores) / len(task_scores)
            metrics["eval_composite_score"] = composite_score

        # Add evaluation loss to metrics
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()


        # Ensure metrics are JSON-serializable
        metrics = denumpify_detensorize(metrics)
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (:obj:`str` or :obj:`bool`, `optional`):
                If a :obj:`str`, local path to a saved checkpoint as saved by a previous instance of
                :class:`~transformers.Trainer`. If a :obj:`bool` and equals `True`, load the last checkpoint in
                `args.output_dir` as saved by a previous instance of :class:`~transformers.Trainer`. If present,
                training will resume from the model/optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if args.fp16_full_eval and not args.do_train:
            self.model = self.model.to(args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed(args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not os.path.isfile(os.path.join(resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

            logger.info(f"Loading model from {resume_from_checkpoint}).")

            if os.path.isfile(os.path.join(resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self.model = self.model.to(args.device)
            self.model_wrapped = self.model

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training datalaoder has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = len(self.train_dataset) * args.num_train_epochs
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = args.max_steps
            num_train_epochs = int(args.num_train_epochs)
            num_update_steps_per_epoch = max_steps
            num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        model = self._wrap_model(self.model_wrapped)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)



        # Train!
        num_examples = (
            self.num_examples(train_dataloader) if train_dataset_is_sized else total_train_batch_size * args.max_steps
        )

        logger.info("***** Running training*****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, "trainer_state.json")
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        if hasattr(self.model, "weighting_method"):
            self.model.weighting_method.train_loss_buffer = torch.zeros(
                (self.model.weighting_method.task_num, num_train_epochs)
            ).to(self.args.device)

        for epoch in range(epochs_trained, num_train_epochs):

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator) if train_dataset_is_sized else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if hasattr(self.model, "weighting_method"):
                self.model.weighting_method.epoch = epoch
                epoch_task_losses = [0.0, 0.0, 0.0, 0.0]
            fs = 0
            for step, inputs in enumerate(epoch_iterator):

                fs = step
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss += self.training_step(model, inputs)
                else:
                    tr_loss += self.training_step(model, inputs)
                self.current_flos += float(self.floating_point_ops(inputs))


                if hasattr(self.model.weighting_method, 'current_step_losses'):
                    current_losses = self.model.weighting_method.current_step_losses
                    for i in range(len(current_losses)):
                        epoch_task_losses[i] += current_losses[i]

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.use_amp:
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.use_amp:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            avg_losses = []
            for i in range(len(current_losses)):
                avg_losses.append(current_losses[i] / fs)

            # 更新loss buffer（所有设备同步更新）
            self.model.weighting_method.train_loss_buffer[:, epoch] = (
                torch.tensor(avg_losses).to(self.args.device)
            )

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )

            best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
            if os.path.exists(best_model_path):
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(best_model_path, map_location="cpu")
                # If the model is on the GPU, it still works!
                self._load_state_dict_in_model(state_dict)
            else:
                logger.warn(
                    f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                    "on multiple nodes, you should activate `--save_on_each_node`."
                )

            if self.deepspeed:
                self.deepspeed.load_checkpoint(
                    self.state.best_model_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False
                )

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
class ZipDataLoader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        print("dataset len: ", [len(dataloader.dataset) for dataloader in self.dataloaders])
        self.dataset = dataloaders[1].dataset
        self.batch_size = dataloaders[0].batch_size

    def __iter__(self):
        iterators = [iter(dl) for dl in self.dataloaders]
        max_length = max(len(dl) for dl in self.dataloaders)
        current_step = 0
        while current_step < max_length:
            batches = []
            for i, it in enumerate(iterators):
                try:
                    batch = next(it)
                except StopIteration:
                    iterators[i] = iter(self.dataloaders[i])
                    batch = next(iterators[i])
                batches.append(batch)
            yield {"batch1": batches[0], "batch2": batches[1], "batch3": batches[2]}
            current_step += 1
            del batches

    def __len__(self):
        result = max(len(dataloader) for dataloader in self.dataloaders)
        return result

