from torch.utils.data import ConcatDataset
from transformers import BartConfig, Seq2SeqTrainingArguments, EarlyStoppingCallback, \
    IntervalStrategy, SchedulerType
import torch.nn as nn
import logging
from typing import Union, Tuple
import os

from models.bart import BartForClassificationAndGeneration,BartForMoe
from data.vocab import Vocab, load_vocab, init_vocab
from data.dataset import init_dataset
from utils.general import count_params, human_format, layer_wise_parameters
from eval.metrics import bleu, meteor, rouge_l, avg_ir_metrics, accuracy_for_sequence, accuracy_top_k_for_sequence
from utils.callbacks import LogStateCallBack
from utils.trainer import CodeTrainer
from utils.trainer import CustomTrainer
import enums
import torch
from tqdm import tqdm
from data.data_collator import collate_fn

logger = logging.getLogger(__name__)


def run_mul_task(
        args,
        trained_model: Union[BartForClassificationAndGeneration, str] = None,
        trained_vocab: Union[Tuple[Vocab, Vocab, Vocab], str] = None,
        only_test=False):
    """
    Fine-tuning from given pre-trained model and vocabs, or training from scratch.

    Args:
        args (argparse.Namespace): Arguments
        trained_model (Union[BartForClassificationAndGeneration, str]): Optional,
            instance or directory of ``BartForClassificationAndGeneration``, must given when ``only_test`` is True
        trained_vocab (Union[Tuple[Vocab, Vocab, Vocab], str]): Optional, Tuple of instances or directory of three
            vocabularies, must given when ``only_test`` is True
        only_test (bool): True when only need to test, default to False

    """
    logger.info('-' * 100)
    logger.info('Start Mul-task')
    # logger.info(f'Code summarization on language: {args.summarization_language}')
    # --------------------------------------------------
    # summarization datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading sum datasets')
    sumDatasets = dict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        sumDatasets[split] = init_dataset(args=args,
                                          mode=enums.TRAINING_MODE_FINE_TUNE,
                                          task=enums.TASK_SUMMARIZATION,
                                          language=args.summarization_language,
                                          split=split)
        logger.info(f'The size of {split} set: {len(sumDatasets[split])}')
    if args.train_subset_ratio and 'train' in sumDatasets:
        sumDatasets['train'] = sumDatasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(sumDatasets['train'])))

    logger.info('sumDatasets loaded successfully')

    # --------------------------------------------------
    # completion datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading com datasets')
    comDatasets = dict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        comDatasets[split] = init_dataset(args=args,
                                          mode=enums.TRAINING_MODE_FINE_TUNE,
                                          task=enums.TASK_COMPLETION,
                                          split=split)
        logger.info(f'The size of {split} set: {len(comDatasets[split])}')
    if args.train_subset_ratio and 'train' in comDatasets:
        comDatasets['train'] = comDatasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(comDatasets['train'])))


    # --------------------------------------------------
    # translation datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading trans datasets')
    transDatasets = dict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        transDatasets[split] = init_dataset(args=args,
                                            mode=enums.TRAINING_MODE_FINE_TUNE,
                                            task=enums.TASK_TRANSLATION,
                                            language=
                                            f'{args.translation_source_language}-{args.translation_target_language}',
                                            split=split)
        logger.info(f'The size of {split} set: {len(transDatasets[split])}')
    if args.train_subset_ratio and 'train' in transDatasets:
        transDatasets['train'] = transDatasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(transDatasets['train'])))
    logger.info('transDatasets loaded successfully')

    # --------------------------------------------------
    # bugfix datasets
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Loading bug_fix datasets')
    bugfixDatasets = dict()
    splits = ['train', 'valid', 'test']
    for split in splits:
        bugfixDatasets[split] = init_dataset(args=args,
                                       mode=enums.TRAINING_MODE_FINE_TUNE,
                                       task=enums.TASK_BUG_FIX,
                                       language=args.bug_fix_scale,
                                       split=split)
        logger.info(f'The size of {split} set: {len(bugfixDatasets[split])}')
    if args.train_subset_ratio and 'train' in bugfixDatasets:
        bugfixDatasets['train'] = bugfixDatasets['train'].subset(args.train_subset_ratio)
        logger.info(f'The train is trimmed to subset due to the argument: train_subset_ratio={args.train_subset_ratio}')
        logger.info('The size of trimmed train set: {}'.format(len(bugfixDatasets['train'])))
    logger.info('Datasets loaded successfully')


    # --------------------------------------------------
    # vocabs
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_vocab:
        if isinstance(trained_vocab, tuple):
            logger.info('Vocabularies are passed through parameter')
            assert len(trained_vocab) == 3
            code_vocab, ast_vocab, nl_vocab = trained_vocab
        else:
            logger.info('Loading vocabularies from files')
            code_vocab = load_vocab(vocab_root=trained_vocab, name=args.code_vocab_name)
            ast_vocab = load_vocab(vocab_root=trained_vocab, name=args.ast_vocab_name)
            nl_vocab = load_vocab(vocab_root=trained_vocab, name=args.nl_vocab_name)
    else:
        logger.info('Building vocabularies')
        code_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                                name=args.code_vocab_name,
                                method=args.code_tokenize_method,
                                vocab_size=args.code_vocab_size,
                                datasets=[comDatasets['train'].codes],
                                ignore_case=True,
                                save_root=args.vocab_root)
        nl_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                              name=args.nl_vocab_name,
                              method=args.nl_tokenize_method,
                              vocab_size=args.nl_vocab_size,
                              datasets=[comDatasets['train'].nls],
                              ignore_case=True,
                              save_root=args.vocab_root,
                              index_offset=len(code_vocab))
        ast_vocab = init_vocab(vocab_save_dir=args.vocab_save_dir,
                               name=args.ast_vocab_name,
                               method='word',
                               datasets=[comDatasets['train'].asts],
                               save_root=args.vocab_root,
                               index_offset=len(code_vocab) + len(nl_vocab))
    logger.info(f'The size of code vocabulary: {len(code_vocab)}')
    logger.info(f'The size of nl vocabulary: {len(nl_vocab)}')
    logger.info(f'The size of ast vocabulary: {len(ast_vocab)}')
    logger.info('Vocabularies built successfully')

    # --------------------------------------------------
    # model
    # --------------------------------------------------
    logger.info('-' * 100)
    if trained_model:
        if isinstance(trained_model, BartForClassificationAndGeneration):
            logger.info('Model is passed through parameter')
            model = trained_model
        else:
            logger.info('Loading the model from file')
            config = BartConfig.from_json_file(os.path.join(trained_model, 'config.json'))
            model = BartForClassificationAndGeneration.from_pretrained(os.path.join(trained_model, 'pytorch_model.bin'),
                                                                       config=config)
    else:
        logger.info('Building the model')
        config = BartConfig(vocab_size=len(code_vocab) + len(nl_vocab) + len(ast_vocab),
                            max_position_embeddings=1024,
                            encoder_layers=args.n_layer,
                            encoder_ffn_dim=args.d_ff,
                            encoder_attention_heads=args.n_head,
                            decoder_layers=args.n_layer,
                            decoder_ffn_dim=args.d_ff,
                            decoder_attention_heads=args.n_head,
                            activation_function='gelu',
                            d_model=args.d_model,
                            dropout=args.dropout,
                            use_cache=True,
                            pad_token_id=Vocab.START_VOCAB.index(Vocab.PAD_TOKEN),
                            bos_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            is_encoder_decoder=True,
                            decoder_start_token_id=Vocab.START_VOCAB.index(Vocab.SOS_TOKEN),
                            forced_eos_token_id=Vocab.START_VOCAB.index(Vocab.EOS_TOKEN),
                            max_length=args.max_nl_len,
                            min_length=1,
                            num_beams=args.beam_width,
                            num_labels=2)
        model = BartForClassificationAndGeneration(config)
        # model = BartForMoe(config)
    model.set_model_mode(enums.MODEL_MODE_GEN)


    # log model statistics
    logger.info('Trainable parameters: {}'.format(human_format(count_params(model))))
    table = layer_wise_parameters(model)
    logger.debug('Layer-wised trainable parameters:\n{}'.format(table))
    logger.info('Model built successfully')

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------
    logger.info('-' * 100)
    logger.info('Initializing the running configurations')

    def decode_preds_1(preds):
        preds, labels = preds
        decoded_preds = nl_vocab.decode_batch(preds)
        decoded_labels = nl_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    def decode_preds_2(preds):
        preds, labels = preds
        decoded_preds = code_vocab.decode_batch(preds)
        decoded_labels = code_vocab.decode_batch(labels)
        return decoded_labels, decoded_preds

    # compute metrics
    def compute_valid_metrics_1(eval_preds):
        # sum
        decoded_labels, decoded_preds = decode_preds_1(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        result.update(rouge_l(references=refs, candidates=cans))
        return result

    def compute_valid_metrics_2(eval_preds):
        # com
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    def compute_valid_metrics_3(eval_preds):
        # trans
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    def compute_valid_metrics_4(eval_preds):
        # bugfix
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result = {}
        result.update(bleu(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    compute_metrics_dict = {
        "batch1": compute_valid_metrics_4,  # 任务 1 的指标计算函数
        "batch2": compute_valid_metrics_2,  # 任务 2 的指标计算函数
        "batch3": compute_valid_metrics_3,  # 任务 3 的指标计算函数
        "batch4": compute_valid_metrics_1
    }

    training_args = Seq2SeqTrainingArguments(output_dir=os.path.join(args.checkpoint_root, enums.TASK_SUMMARIZATION),
                                             overwrite_output_dir=True,
                                             do_train=True,
                                             do_eval=True,
                                             do_predict=True,
                                             evaluation_strategy=IntervalStrategy.EPOCH,
                                             prediction_loss_only=False,
                                             per_device_train_batch_size=args.batch_size,
                                             per_device_eval_batch_size=args.eval_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             learning_rate=args.learning_rate,
                                             weight_decay=args.lr_decay_rate,
                                             max_grad_norm=args.grad_clipping_norm,
                                             num_train_epochs=args.n_epoch,
                                             lr_scheduler_type=SchedulerType.LINEAR,
                                             warmup_steps=args.warmup_steps,
                                             logging_dir=os.path.join(args.tensor_board_root, enums.TASK_SUMMARIZATION),
                                             logging_strategy=IntervalStrategy.STEPS,
                                             logging_steps=args.logging_steps,
                                             save_strategy=IntervalStrategy.EPOCH,
                                             save_total_limit=5,
                                             seed=args.random_seed,
                                             fp16=args.fp16,
                                             dataloader_drop_last=False,
                                             run_name=args.model_name,
                                             load_best_model_at_end=True,
                                             metric_for_best_model='composite_score',
                                             greater_is_better=True,
                                             ignore_data_skip=False,
                                             label_smoothing_factor=args.label_smoothing,
                                             report_to=['tensorboard'],
                                             dataloader_pin_memory=True,
                                             predict_with_generate=True)
    datasets = dict()
    datasets['train'] = [bugfixDatasets['train'], comDatasets['train'], transDatasets['train'], sumDatasets['train']]
    datasets['valid'] = [bugfixDatasets['valid'], comDatasets['valid'], transDatasets['valid'], sumDatasets['valid']]
    datasets['test'] = [bugfixDatasets['test'], comDatasets['test'], transDatasets['test'], sumDatasets['test']]

    # datasets['train'] = ConcatDataset([sumDatasets['train'],transDatasets['train']])
    # datasets['valid'] = ConcatDataset([sumDatasets['valid'],transDatasets['valid']])
    # datasets['test'] = [sumDatasets['test'], comDatasets['test'], transDatasets['test']]
    # logger.info("here!!!!")
    # logger.info([len(e) for e in datasets["train"]])
    trainer = CustomTrainer(main_args=args,
                            code_vocab=code_vocab,
                            ast_vocab=ast_vocab,
                            nl_vocab=nl_vocab,
                            task=enums.TASK_SUMMARIZATION,
                            model=model,
                            args=training_args,
                            data_collator=None,
                            # 修改
                            train_datasets=datasets['train'] if 'train' in datasets else None,
                            eval_datasets=datasets['valid'] if 'valid' in datasets else None,
                            tokenizer=nl_vocab,
                            model_init=None,
                            compute_metrics=compute_valid_metrics_2,
                            # 修改
                            compute_metrics_dict=compute_metrics_dict,
                            callbacks=[
                                EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                                LogStateCallBack()],
                            weighting_method='dwa')

    # # 获取训练 DataLoader
    # train_dataloader = trainer.get_train_dataloader()
    #
    # for step, inputs in enumerate(train_dataloader):
    #     # 遍历每个任务
    #     for task_key, batch in inputs.items():
    #         print(task_key)
    #         # 查看每个值的形状
    #         for key, value in batch.items():
    #             print(f"{key}: {value}")
    #     if step == 1:
    #         break
    #
    # # 获取验证 DataLoader
    # eval_dataloader = trainer.get_eval_dataloader(eval_dataset=None)
    #
    # # 遍历 DataLoader 并打印数据
    # for step, inputs in enumerate(eval_dataloader):
    #     # 遍历每个任务
    #     for task_key, batch in inputs.items():
    #         print(task_key)
    #         # 查看每个值的形状
    #         for key, value in batch.items():
    #             print(f"{key}: {value}")
    #     if step == 1:
    #         break
    # exit()
    logger.info('Running configurations initialized successfully')

    # --------------------------------------------------
    # train
    # --------------------------------------------------
    if not only_test:
        logger.info('-' * 100)
        logger.info('Start training')
        # 检查点
        # resume_from_checkpoint = '../outputs/default_model_20250226_064418/checkpoints/summarization/checkpoint-39096/'
        train_result = trainer.train()
        logger.info('Training finished')
        trainer.save_model(args.model_root)
        trainer.save_state()
        metrics = train_result.metrics
        trainer.log_metrics(split='train', metrics=metrics)
        trainer.save_metrics(split='train', metrics=metrics)

        # --------------------------------------------------
        # eval
        # --------------------------------------------------
        # logger.info('-' * 100)
        # logger.info('Start evaluating')
        # eval_metrics = trainer.evaluate(metric_key_prefix='valid',
        #                                 max_length=args.max_decode_step,
        #                                 num_beams=args.beam_width)
        # trainer.log_metrics(split='valid', metrics=eval_metrics)
        # trainer.save_metrics(split='valid', metrics=eval_metrics)

    # --------------------------------------------------
    # predict
    # --------------------------------------------------
    trainer = CodeTrainer(main_args=args,
                          code_vocab=code_vocab,
                          ast_vocab=ast_vocab,
                          nl_vocab=nl_vocab,
                          task=enums.TASK_SUMMARIZATION,
                          model=model,
                          args=training_args,
                          data_collator=None,
                          train_dataset=datasets['train'] if 'train' in datasets else None,
                          eval_dataset=datasets['valid'] if 'valid' in datasets else None,
                          tokenizer=nl_vocab,
                          model_init=None,
                          compute_metrics=compute_valid_metrics_1,
                          callbacks=[
                              EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience),
                              LogStateCallBack()])
    # Summarization
    logger.info('-' * 100)
    logger.info('Start testing summarization')
    # model.set_id(3)
    def compute_test_metrics_1(eval_preds):
        decoded_labels, decoded_preds = decode_preds_1(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        # result.update(meteor(references=refs, candidates=cans))
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    trainer.compute_metrics = compute_test_metrics_1
    predict_results = trainer.predict(test_dataset=sumDatasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_nl_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_SUMMARIZATION}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_SUMMARIZATION}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_SUMMARIZATION}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Summarization testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')


    # Completion
    logger.info('-' * 100)
    logger.info('Start testing completion')
    # model.set_id(1)
    def compute_test_metrics_2(eval_preds):
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        try:
            result.update(meteor(references=refs, candidates=cans))
        except Exception:
            pass
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    trainer.task = enums.TASK_COMPLETION
    trainer.compute_metrics = compute_test_metrics_2
    predict_results = trainer.predict(test_dataset=comDatasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_code_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_COMPLETION}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_COMPLETION}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_COMPLETION}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')

    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')

    logger.info('-' * 100)
    logger.info('Start testing accuracy at 5')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    torch.cuda.empty_cache()
    test_dataloader = torch.utils.data.DataLoader(dataset=comDatasets['test'],
                                                  batch_size=args.eval_batch_size,
                                                  collate_fn=lambda batch: collate_fn(batch,
                                                                                      args=args,
                                                                                      task=enums.TASK_COMPLETION,
                                                                                      code_vocab=code_vocab,
                                                                                      nl_vocab=nl_vocab,
                                                                                      ast_vocab=ast_vocab))
    predictions = []
    references = []
    for step, batch in enumerate(tqdm(test_dataloader)):
        batch_size = batch['input_ids'].size(0)
        batch_outputs = model.generate(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            max_length=args.completion_max_len,
            min_length=3,
            early_stopping=True,
            num_beams=args.beam_width,
            num_return_sequences=5
        )
        batch_outputs = batch_outputs.view(batch_size, -1, batch_outputs.size(-1))
        for outputs in batch_outputs:
            decoded = code_vocab.decode_batch(outputs.cpu().numpy())
            predictions.append(decoded)

        labels = code_vocab.decode_batch(batch['labels'].numpy())
        references += labels

    assert len(predictions) == len(references)
    scores = accuracy_top_k_for_sequence(references=references, candidates=predictions)

    for name, score in scores.items():
        logger.info(f'{name}: {score}')

    with open(os.path.join(args.output_root, f'{enums.TASK_COMPLETION}_test_top_k_results.txt'),
              mode='w',
              encoding='utf-8') as f:
        sample_id = 0
        for reference, candidate in zip(references, predictions):
            f.write(f'sample {sample_id}:\n')
            f.write(f'reference: {reference}\n')
            for idx, can in enumerate(candidate):
                f.write(f'candidate {idx}: {can}\n')
            f.write('\n')
            sample_id += 1
        for name, score in scores.items():
            f.write(f'{name}: {score}')

    logger.info('Completion testing finished')

    # Translation
    logger.info('-' * 100)
    logger.info('Translation start testing')
    # model.set_id(2)
    def compute_test_metrics_3(eval_preds):
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        try:
            result.update(meteor(references=refs, candidates=cans))
        except Exception:
            pass
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    trainer.task = enums.TASK_TRANSLATION
    trainer.compute_metrics = compute_test_metrics_3
    predict_results = trainer.predict(test_dataset=transDatasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_code_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_TRANSLATION}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_TRANSLATION}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_TRANSLATION}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Translation testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')

    # Bugfix
    logger.info('-' * 100)
    logger.info('Start testing bugfix')
    # model.set_id(0)
    def compute_test_metrics_4(eval_preds):
        decoded_labels, decoded_preds = decode_preds_2(eval_preds)
        result = {'references': decoded_labels, 'candidates': decoded_preds}
        refs = [ref.strip().split() for ref in decoded_labels]
        cans = [can.strip().split() for can in decoded_preds]
        result.update(bleu(references=refs, candidates=cans))
        try:
            result.update(meteor(references=refs, candidates=cans))
        except Exception:
            pass
        result.update(rouge_l(references=refs, candidates=cans))
        result.update(avg_ir_metrics(references=refs, candidates=cans))
        result.update(accuracy_for_sequence(references=refs, candidates=cans))
        return result

    trainer.task = enums.TASK_BUG_FIX
    trainer.compute_metrics = compute_test_metrics_4
    predict_results = trainer.predict(test_dataset=bugfixDatasets['test'],
                                      metric_key_prefix='test',
                                      max_length=args.max_code_len,
                                      num_beams=args.beam_width)
    predict_metrics = predict_results.metrics
    references = predict_metrics.pop('test_references')
    candidates = predict_metrics.pop('test_candidates')
    trainer.log_metrics(split='test', metrics=predict_metrics)
    trainer.save_metrics(split='test', metrics=predict_metrics)
    # save testing results
    with open(os.path.join(args.output_root, f'{enums.TASK_BUG_FIX}_test_results.txt'),
              mode='w', encoding='utf-8') as result_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_BUG_FIX}_test_refs.txt'),
                 mode='w', encoding='utf-8') as refs_f, \
            open(os.path.join(args.output_root, f'{enums.TASK_BUG_FIX}_test_cans.txt'),
                 mode='w', encoding='utf-8') as cans_f:
        sample_id = 0
        for reference, candidate in zip(references, candidates):
            result_f.write(f'sample {sample_id}:\n')
            sample_id += 1
            result_f.write(f'reference: {reference}\n')
            result_f.write(f'candidate: {candidate}\n')
            result_f.write('\n')
            refs_f.write(reference + '\n')
            cans_f.write(candidate + '\n')
        for name, score in predict_metrics.items():
            result_f.write(f'{name}: {score}\n')
    logger.info('Bugfix testing finished')
    for name, score in predict_metrics.items():
        logger.info(f'{name}: {score}')

    logger.info('All finished!')


