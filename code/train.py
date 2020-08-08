import argparse
import copy
import importlib
import json
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument(
    "config",
    help="Path to experiment configuration file"
)
parser.add_argument(
    "--clear",
    "-c",
    help="If provided remove all saved models and results "
         "and redo experiment.",
    action="store_true"
)
args = parser.parse_args()


def get_nested_dict_elem(d, path):
    for k in path:
        d = d[k]
    return d


def set_nested_dict_elem(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def find_varied_params(params, config, path):
    sub_dict = get_nested_dict_elem(config, path)
    if "type" in sub_dict and sub_dict["type"] == "varied":
        name = sub_dict["name"] if "name" in sub_dict else path[-1]
        if name in params and "main" not in sub_dict:
            raise ValueError(
                f'There are 2 varied params with identical names "{name}"')
        if name not in params:
            params[name] = {}
        if "name" not in sub_dict:
            params[name] = {"main": path}
        params[name][path] = sub_dict["values"]
    else:
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                find_varied_params(params, config, path + (k,))


def check_path_is_not_varied(config, path_key):
    if path_key not in config:
        return
    if not isinstance(config[path_key], dict):
        return
    if "type" in config[path_key] and config[path_key]["type"] == "varied":
        raise ValueError(
            'Parameter "{path_key}" cannot be varied. There can be only one '
            'save path for a configuration path')


def expand_param_variations(config):
    check_path_is_not_varied(config["train"], "results_save_path")
    check_path_is_not_varied(config["train"], "model_save_path")
    configs, param_values_by_config = [config], []
    varied_params = {}
    # fills `varied_params` dictionary
    find_varied_params(varied_params, config, path=())
    for param_name, varied_specs in varied_params.items():
        paths_and_values = [
            (p, v) for p, v in varied_specs.items() if isinstance(p, tuple)]
        paths, values = zip(*paths_and_values)
        main_path_idx = paths.index(varied_specs["main"])
        new_configs, new_param_values_by_config = [], []
        for c, c_p in zip(configs, param_values_by_config):
            for set_idx, one_values_set in enumerate(zip(values)):
                new_c = copy.deepcopy(c)
                new_c_p = copy.deepcopy(c_p)
                new_c_p[param_name] = one_values_set[main_path_idx]
                for i, v in one_values_set:
                    set_nested_dict_elem(new_c, paths[i], v)
                new_configs.append(new_c)
                new_param_values_by_config.append(new_c_p)
        configs = new_configs
        param_values_by_config = new_param_values_by_config
    return configs, param_values_by_config


def get_root_path(config) -> Path:
    default_root_path = "~/corr-paper"
    if "vars" in config:
        root_path = Path(config.get("vars").get("ROOT", default_root_path))
    else:
        root_path = Path(default_root_path)
    return root_path.expanduser()


def set_save_paths(config):
    root_path = get_root_path(config)
    try:
        config_rel_path = \
            Path(__file__).relative_to(root_path / Path('code/configs'))
    except ValueError:
        if "result_save_path" not in config["train"]:
            raise ValueError(
                'If configuration file is not in \'{ROOT}/code/configs\' than '
                'you have to provide "result_save_path" in "train" section of '
                'config.'
            )
        if "model_save_path" not in config["train"]:
            raise ValueError(
                'If configuration file is not in \'{ROOT}/code/configs\' than '
                'you have to provide "model_save_path" in "train" section of'
                ' config.')
        return
    if "result_save_path" not in config["train"]:
        config["train"]["result_save_path"] = \
            root_path / Path("results") / config_rel_path
    if "model_save_path" not in config["train"]:
        config["train"]["model_save_path"] = \
            root_path / Path("code/configs") / config_rel_path


def import_object(cls_descr):
    elems = cls_descr.split('.')
    module_name = '.'.join(elems[:-1])
    old_sys_path_0 = sys.path[0]
    sys.path[0] = Path(__file__).parent
    module = importlib.import_module(module_name)
    sys.path[0] = old_sys_path_0
    return getattr(module, elems[-1])


def build_iterator(config):
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(**config)


def build_model(config):
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(**config)


def get_lr_decay_method(config):
    if 'lr_step' in config:
        method_of_lr_decay = 'periodic'
    elif 'lr_patience' in config and 'lr_patience_period' in config:
        method_of_lr_decay = 'impatience'
    else:
        raise ValueError(
            "Train config has to contain either parameter 'lr_step' or "
            "parameters 'lr_patience' and 'lr_patience_period'.\n"
            "train config={}".format(config)
        )
    return method_of_lr_decay


def update_lr(lr, step, valid_metrics, best_ce_loss, lr_impatience, config):
    method_of_lr_decay = get_lr_decay_method(config)
    if method_of_lr_decay == 'periodic':
        lr = config['lr_init'] \
             * config['lr_decay'] ** (step // config["lr_step"])
    elif method_of_lr_decay == 'impatience':
        if step % config['lr_patience_period'] == 0:
            if valid_metrics['ce_loss'] < best_ce_loss:
                lr_impatience = 0
                best_ce_loss = valid_metrics['ce_loss']
            else:
                lr_impatience += 1
                if lr_impatience > config['lr_patience']:
                    lr *= config['lr_decay']
                    lr_impatience = 0
    return lr, lr_impatience, best_ce_loss


def get_training_interruption_method(config):
    if 'num_steps' in config:
        method_of_interrupting_training = 'fixed_num_steps'
    elif 'stop_patience_period' in config and 'stop_patience' in config:
        method_of_interrupting_training = 'impatience'
    else:
        raise ValueError(
            "Train config has to contain either parameter 'fixed_num_steps'"
            "or parameters 'stop_patience' and 'stop_patience_period'.\n"
            "train config={}".format(config)
        )
    return method_of_interrupting_training


def decide_if_training_is_finished(
        step, valid_metrics, best_ce_loss, stop_impatience, config):
    method_of_interruption_of_training = get_training_interruption_method(
        config)
    if method_of_interruption_of_training == 'fixed_num_steps':
        stop_training = step > config['num_steps']
    elif method_of_interruption_of_training == 'impatience':
        if step % config['stop_patience_period'] == 0:
            if valid_metrics['ce_loss'] < best_ce_loss:
                stop_impatience = 0
                best_ce_loss = valid_metrics['ce_loss']
            else:
                stop_impatience += 1
        stop_training = stop_impatience > config['stop_patience']
    else:
        raise ValueError(
            "Unsupported method of interrupting training '{}'".format(
                method_of_interruption_of_training
            )
        )
    return stop_training, stop_impatience, best_ce_loss


def time_for_logarithmic_logging(step, factor):
    if step == 0:
        return True
    step_is_integer_power_of_factor = int(np.log(step+1) / np.log(factor)) \
        - int(np.log(step) / np.log(factor)) > 0
    return step_is_integer_power_of_factor


def compute_metrics(pred_probas, labels, metric_specs):
    pass  # TODO


def save_metrics_and_params(
        step, data_type, metric_values, lr, additional_measurements):
    pass  # TODO


def append_step_metrics(accumulated_metrics, step_metrics):
    pass  # TODO


def average_metrics(accumulated_metrics):
    pass  # TODO


def post_process_hooks(model, hook_post_processing):
    pass  # TODO


def test(
        iterator,
        model,
        data_type,
        config
):
    loss_cls = import_object(config["train"]["loss_cls"])
    loss_fn = loss_cls()
    model.eval()
    required_metrics = config.get("metrics", {})
    accumulated_metrics = {'loss': []}
    for k in required_metrics:
        accumulated_metrics[k] = []
    for test_step, (inputs, labels) in enumerate(
            iterator.gen_batches(data_type, config[data_type]["batch_specs"])):
        pred_probas = model(inputs)
        step_metrics = compute_metrics(pred_probas, labels, required_metrics)
        accumulated_metrics = append_step_metrics(
            accumulated_metrics, step_metrics)
        accumulated_metrics['loss'].append(loss_fn(pred_probas, labels))

    metrics = average_metrics(accumulated_metrics)
    additional_measurements = post_process_hooks(
        model, config['valid']['hooks_post_processing_fns'])
    return metrics, additional_measurements


def log(step, data_type, metric_values, lr):
    pass  # TODO


def train(config, iterator, model, varied_params, config_idx, repeat_idx):
    optimizer_cls = import_object(config["train"]["optimizer_cls"])
    loss_cls = import_object(config["train"]["loss_cls"])
    loss_fn = loss_cls()

    lr = config['lr_init']

    stop_impatience = 0
    # `lr_impatience` is not used if 'lr_step' is in `config`.
    lr_impatience = 0
    best_stop_ce_loss = float('+inf')
    best_lr_ce_loss = float('+inf')

    for step, (inputs, labels) in enumerate(
            iterator.gen_batches(
                "train", config["batch_specs"], infinite=True)):
        if time_for_logarithmic_logging(step, config['log_factor']):
            v_metrics = test(
                step, iterator, model, "valid", config)
            save_metrics_and_params(step, 'valid', v_metrics, lr,
                model.post_processed_accumulator_hooks)
            log(step, 'valid', v_metrics, lr)
            if step > 0:
                save_metrics_and_params(
                    step, 'train', t_metrics, lr, model.last_run_hook_values)
                log(step, 'train', t_metrics, lr)
        model.train()
        pred_probas = model(inputs)
        loss = loss_fn(pred_probas, labels)
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        loss.backward()
        optimizer.step()

        t_metrics = compute_metrics(
            pred_probas, labels, config.get("metrics", {}))
        t_metrics['loss'] = loss

        lr, lr_impatience, best_lr_ce_loss = update_lr(
            lr, step, v_metrics, best_lr_ce_loss, lr_impatience, config)

        stop_training, stop_impatience, best_stop_ce_loss = \
            decide_if_training_is_finished(
                step, v_metrics, best_stop_ce_loss, stop_impatience, config)
        if stop_training:
            break


def build_and_run_once(config, varied_params, config_idx, repeat_idx):
    iterator = build_iterator(copy.deepcopy(config["iterator"]))
    model_config = copy.deepcopy(config["model"])
    model_config["element_shape"] = iterator.element_shape
    model = build_model(model_config)
    train(
        config["train"],
        iterator,
        model,
        varied_params,
        config_idx,
        repeat_idx
    )


def build_and_run_repeatedly(config, varied_params, config_idx):
    for repeat_idx in range(config["num_repeats"]):
        build_and_run_once(config, varied_params, config_idx, repeat_idx)


def main():
    with open(args.config) as f:
        config = json.load(f)
    set_save_paths(config)
    configs, param_values_by_config = expand_param_variations(config)
    for i, (varied_params, config) in enumerate(
            zip(param_values_by_config, configs)):
        with open(
                Path(config["train"]["result_save_path"]).expanduser()
                / Path(f"param_sets/{i}.json"), "w") \
                as f:
            json.dump(varied_params, f)
        build_and_run_repeatedly(config, varied_params, i)


if __name__ == '__main__':
    main()
