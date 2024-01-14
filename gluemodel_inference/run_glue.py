# coding=utf-8

import faulthandler
faulthandler.enable()
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    glue_compute_metrics,
    glue_output_modes,
    glue_tasks_num_labels,
    set_seed,
)

from length_adaptive_transformer import (
    TrainingArguments,
    LengthDropArguments,
    SearchArguments,
    LengthDropTrainer,
)
from length_adaptive_transformer.drop_and_restore_utils import (
    sample_length_configuration,
)
from length_adaptive_transformer.evolution import (
    approx_ratio, inverse, store2str
)

logger = logging.getLogger(__name__)

glue_tasks_metrics = {
    "cola": "mcc",
    "mnli": "mnli/acc",
    "mnli-mm": "mnli-mm/acc",
    "mrpc": "acc",
    "sst-2": "acc",
    "sts-b": "spearmanr",
    "qqp": "acc",
    "qnli": "acc",
    "rte": "acc",
    "wnli": "acc",
}



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class LengthAdaptiveArguments:
    model_type: str = field(default=None)
    eval_length_config: str = field(default=None)
    eval_lengthdrop: bool = False
    avg_time: int = field(default=50)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, LengthDropArguments, SearchArguments,LengthAdaptiveArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, length_drop_args, search_args, la_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, length_drop_args, search_args, la_args = parser.parse_args_into_dataclasses()

    if 'sst' in data_args.task_name or 'SST' in data_args.task_name:
        data_args.task_name = 'sst-2'

    if 'stsb' in data_args.task_name:
        data_args.task_name = 'sts-b'

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    # logging.basicConfig(
    #     # format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     format="%(asctime)s: %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    # )
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     training_args.local_rank,
    #     training_args.device,
    #     training_args.n_gpu,
    #     bool(training_args.local_rank != -1),
    #     training_args.fp16,
    # )
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # print(config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    import copy
    if data_args.task_name == 'MNLI' or data_args.task_name == 'mnli':
        if not ('bert' in model_args.model_name_or_path or 'BERT' in model_args.model_name_or_path):
            tmp_w = copy.deepcopy(model.classifier.out_proj.weight.data)
            tmp_b = copy.deepcopy(model.classifier.out_proj.bias.data)
            a, b, c = 2, 1, 0
            model.classifier.out_proj.weight.data[0, :] = tmp_w[a, :]
            model.classifier.out_proj.weight.data[1, :] = tmp_w[b, :]
            model.classifier.out_proj.weight.data[2, :] = tmp_w[c, :]
            model.classifier.out_proj.bias.data[0] = tmp_b[a]
            model.classifier.out_proj.bias.data[1] = tmp_b[b]
            model.classifier.out_proj.bias.data[2] = tmp_b[c]
        else:
            tmp_w = copy.deepcopy(model.classifier.weight.data)
            tmp_b = copy.deepcopy(model.classifier.bias.data)
            a, b, c = 2,0,1
            model.classifier.weight.data[0, :] = tmp_w[a, :]
            model.classifier.weight.data[1, :] = tmp_w[b, :]
            model.classifier.weight.data[2, :] = tmp_w[c, :]
            model.classifier.bias.data[0] = tmp_b[a]
            model.classifier.bias.data[1] = tmp_b[b]
            model.classifier.bias.data[2] = tmp_b[c]

    # Get datasets
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
            else:  # regression
                preds = np.squeeze(preds)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn


    if search_args.do_search:
        model.config.output_attentions = True

    assert not (length_drop_args.length_config and length_drop_args.length_adaptive)
    if length_drop_args.length_adaptive or search_args.do_search:
        training_args.max_seq_length = data_args.max_seq_length

    # Initialize our Trainer
    trainer = LengthDropTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        best_metric=glue_tasks_metrics[data_args.task_name],
        length_drop_args=length_drop_args,
    )

    # Training
    if training_args.do_train:
        global_step, best = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        best_msg = ", ".join([f"{k} {v}" for k, v in best.items()])
        logger.info(f" global_step = {global_step} | best: {best_msg}")
        '''
        output_dir = os.path.join(training_args.output_dir, "checkpoint-last")
        trainer.save_model(output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(output_dir)
        '''



    la_args.model_type = 'roberta' if config.vocab_size == 50265 else 'bert'
    if la_args.eval_length_config is not None:
        eval_length_config = [int(_) for _ in la_args.eval_length_config.split(',')]
        la_args.eval_length_config = tuple(eval_length_config)

    # Evaluation
    eval_results = {}
    if training_args.do_eval and not search_args.do_search:
        print("*** Evaluate on Dev ***")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            eval_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
            )

        for _, eval_dataset in enumerate(eval_datasets):
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            timecosts = []
            # for i in range(la_args.avg_time + 1):
            for i in range(1):
                eval_result, timecost = trainer.evaluate(eval_dataset=eval_dataset,
                                                         model_type=la_args.model_type,
                                                         length_config=la_args.eval_length_config)
                if i > 0:
                    timecosts.append(timecost)
                output_eval_file = os.path.join(
                    training_args.output_dir, f"eval_results_{eval_dataset.args.task_name}.txt"
                )
                eval_results.update(eval_result)
            if trainer.is_world_master():
                if data_args.task_name == "mnli":
                    if _ == 0:
                        print('On MNLI-m')
                    else:
                        print('On MNLI-mm')
                for key, value in eval_result.items():
                    print("  %s = %s" % (key, value))
            #     with open(output_eval_file, "w") as writer:
            #         for key, value in eval_result.items():
            #             print("  %s = %s" % (key, value))
            #             writer.write("%s = %s\n" % (key, value))
            # timecosts = np.array(timecosts)
            # print('###  MEAN TIME COST OVER %d RUNS ###' % la_args.avg_time)
            # print('### Mean time for all task data (%d examples): %.6f s' % (len(eval_dataset), np.mean(timecosts)))
            # print('### Mean time per example: %.6f s' % (np.mean(timecosts) / len(eval_dataset)))
            # print('### Var: %.6f' % np.var(timecosts, ddof=1))
            # print('### Std: %.6f' % np.std(timecosts, ddof=1))

    if training_args.do_predict:
        print("\n*** Test ***")
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            mnli_mm_data_args = dataclasses.replace(data_args, task_name="mnli-mm")
            test_datasets.append(
                GlueDataset(mnli_mm_data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
            )

        for _, test_dataset in enumerate(test_datasets):
            timecosts = []
            for i in range(la_args.avg_time + 1):
                predictions, timecost = trainer.predict(test_dataset=test_dataset,
                                              model_type=la_args.model_type,
                                              length_config=la_args.eval_length_config
                                              )
                predictions = predictions.predictions
                if i > 0:
                    timecosts.append(timecost)
            timecosts = np.array(timecosts)
            if data_args.task_name == "mnli":
                if _ == 0:
                    print('On MNLI-m')
                else:
                    print('On MNLI-mm')
            print('###  MEAN TIME COST OVER %d RUNS ###' % la_args.avg_time)
            print('### Mean time for all task data (%d examples): %.6f s' % (len(test_dataset), np.mean(timecosts)))
            print('### Mean time per example: %.6f s' % (np.mean(timecosts) / len(test_dataset)))
            # print('### Var: %.6f' % np.var(timecosts, ddof=1))
            # print('### Std: %.6f' % np.std(timecosts, ddof=1))

            if output_mode == "classification":
                predictions = np.argmax(predictions, axis=1)
            if test_dataset.args.task_name == 'mnli-mm':
                output_test_file = os.path.join(
                    training_args.output_dir, 'MNLI-mm.tsv'
                )
            elif test_dataset.args.task_name == 'mnli':
                output_test_file = os.path.join(
                    training_args.output_dir, 'MNLI-m.tsv'
                )
            else:
                output_test_file = os.path.join(
                    training_args.output_dir, data_args.data_dir.split('/')[-1] + '.tsv'
                )
            if trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if output_mode == "regression":
                            writer.write("%d\t%3.3f\n" % (index, item))
                        else:
                            item = test_dataset.get_labels()[item]
                            writer.write("%d\t%s\n" % (index, item))

    # Search
    if search_args.do_search:
        import warnings
        warnings.filterwarnings("ignore")

        output_dir = training_args.output_dir

        # assert args.population_size == args.parent_size + args.mutation_size + args.crossover_size
        trainer.init_evolution()
        trainer.load_store(os.path.join(model_args.model_name_or_path, 'store.tsv'))

        lower_gene = sample_length_configuration(
            data_args.max_seq_length,
            config.num_hidden_layers,
            length_drop_ratio=length_drop_args.length_drop_ratio_bound,
        )
        lower_gene = revise_gene(lower_gene)

        upper_bound = 1.0
        upper_gene = (data_args.max_seq_length, ) * (config.num_hidden_layers)

        trainer.add_gene(lower_gene, method=0)
        # trainer.add_gene((data_args.max_seq_length / 4, ) * (config.num_hidden_layers), method=0)
        trainer.add_gene(upper_gene, method=0)

        trainer.lower_constraint = trainer.store[lower_gene][0]
        trainer.upper_constraint = trainer.store[upper_gene][0]

        length_drop_ratios = [inverse(r) for r in np.linspace(approx_ratio(length_drop_args.length_drop_ratio_bound), upper_bound, search_args.population_size + 2)[1:-1]]

        for p in length_drop_ratios:
            gene = sample_length_configuration(
                data_args.max_seq_length,
                config.num_hidden_layers,
                length_drop_ratio=p,
            )
            gene = revise_gene(gene)
            trainer.add_gene(gene, method=0)

        for i in range(search_args.evo_iter + 1):
            logger.info(f"| Start Iteratsion {i}:")
            print('pareto_frontier')
            population, area = trainer.pareto_frontier()
            print('convex_hull')
            parents = trainer.convex_hull()
            results = {"area": area, "population_size": len(population), "num_parents": len(parents)}

            logger.info(f"| >>>>>>>> {' | '.join([f'{k} {v}' for k, v in results.items()])}")
            for gene in parents:  # population
                logger.info("| " + store2str(gene, *trainer.store[gene][:3]))

            trainer.save_store(os.path.join(output_dir, f'store-iter{i}.tsv'))
            trainer.save_population(os.path.join(output_dir, f'population-iter{i}.tsv'), population)
            trainer.save_population(os.path.join(output_dir, f'parents-iter{i}.tsv'), parents)

            if i == search_args.evo_iter:
                break

            k = 0
            while k < search_args.mutation_size:
                if trainer.mutate(search_args.mutation_prob):
                    k += 1

            k = 0
            while k < search_args.crossover_size:
                if trainer.crossover():
                    k += 1

    return eval_results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def revise_gene(gene):
    gene_new = tuple([_ for _ in gene[:-1]] + [2])
    return gene_new

if __name__ == "__main__":
    main()

