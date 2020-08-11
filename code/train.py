import argparse
import copy
import importlib
import json
import shutil
import sqlite3
import sys
import warnings
from pathlib import Path
from torch.utils.data import DataLoader

import numpy as np


PYTHON_TYPE_TO_SQL_TYPE ={
    int: 'INTEGER',
    float: 'REAL',
    str: 'TEXT'
}


COLUMN_ORDER_IN_DATABASE = [
    "config_idx",
    "repeat_idx",
    "step",
    "learning_rate",
    "loss"
]
# These are followed by metrics in alphabetical order, hooks in alphabetical
# order, and varied params in alphabetical order.


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
        if "main" not in sub_dict:
            is_main = True
        else:
            is_main = sub_dict["main"]
        if name not in params:
            params[name] = {}
        if is_main and "main" in params:
            raise ValueError(
                f'There are 2 varied params with identical names "{name}". '
                f'You probably forgotten to provide `"main" = False` flag to '
                f'one of them.')
        if is_main:
            params[name]["main"] = path
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
    default_root_path = "~/neuron-correlation-decorrelation-paper"
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
            root_path / Path("results") / config_rel_path.with_suffix(".db")
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


def instantiate_object_from_config(config, args=()):
    config = copy.deepcopy(config)
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(*args, **config)


def build_iterator(config):
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(**config)


def build_model(config):
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(**config)


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


def save_metrics_hooks_varied(
        database_path,
        table_name,
        config_idx,
        repeat_idx,
        step,
        learning_rate,
        loss,
        metric_values,
        hooks,
        varied_params,
):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    col_names = cursor.fetchone()
    print("(train.save_metrics_hooks_varied)col_names:", col_names)
    data = {"config_idx": config_idx, "repeat_idx": repeat_idx, "step": step,
            "learning_rate": learning_rate, "loss": loss}
    data.update(metric_values)
    data.update(hooks)
    data.update(varied_params)
    values = []
    for col_name in col_names:
        values.append(data[col_name])
    cursor.execute(f'INSERT INTO {table_name} VALUES {tuple(values)}')
    conn.commit()
    conn.close()

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
    accumulated_metrics = {k: [] for k in required_metrics}
    accumulated_loss = []
    for test_step, (inputs, labels) in enumerate(
            iterator.gen_batches(data_type, config[data_type]["batch_specs"])):
        pred_probas = model(inputs)
        step_metrics = compute_metrics(pred_probas, labels, required_metrics)
        accumulated_metrics = append_step_metrics(
            accumulated_metrics, step_metrics)
        accumulated_loss.append(loss_fn(pred_probas, labels))

    metrics = average_metrics(accumulated_metrics)
    accumulated_loss = sum(accumulated_loss) / len(accumulated_loss)
    hook_values = post_process_hooks(
        model, config['valid']['hooks_post_processing_fns'])
    return metrics, accumulated_loss, hook_values


def log(step, data_type, loss, metric_values, lr):
    pass  # TODO


def train(config, iterator, model, varied_params, config_idx, repeat_idx):
    optimizer = instantiate_object_from_config(
        config["train"]["optimizer"], (model.parameters(),))
    loss_fn = instantiate_object_from_config(config["train"]["loss"])
    scheduler = instantiate_object_from_config(
        config["train"]["scheduler"], (optimizer,))

    stop_impatience = 0
    best_stop_ce_loss = float('+inf')
    trainset = instantiate_object_from_config(config["dataset_reader"])
    trainloader = DataLoader(trainset, batch_size=config["train"]["batch_specs"]["batch_size"], shuffle=True, num_workers=2)

    for step, (inputs, labels) in enumerate(trainloader, 0):
        if time_for_logarithmic_logging(step, config['log_factor']):
            v_metrics, v_loss, v_hooks = test(
                iterator, model, "valid", config)
            save_metrics_hooks_varied(
                config["train"]["result_save_path"], 'valid', config_idx,
                repeat_idx, step, lr, v_loss, v_metrics, v_hooks, varied_params
            )
            log(step, 'valid', v_loss, v_metrics, lr)
            if step > 0:
                save_metrics_hooks_varied(
                    config["train"]["result_save_path"], 'train', config_idx,
                    repeat_idx, step, lr, t_loss, t_metrics, {}, varied_params)
                log(step, 'train', t_loss, t_metrics, lr)
        model.train()
        optimizer.zero_grad()
        pred_probas = model(inputs)
        t_loss = loss_fn(pred_probas, labels)
        t_loss.backward()
        optimizer.step()
        scheduler.step()

        t_metrics = compute_metrics(
            pred_probas, labels, config.get("metrics", {}))
        t_metrics['loss'] = loss

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


def clear(config):
    if "result_save_path" in config["train"]:
        shutil.rmtree(config["train"]["result_save_path"])
    if "model_save_path" in config["train"]:
        shutil.rmtree(config["train"]["model_save_path"])


def create_table(database_file_name, table_name, columns):
    conn = sqlite3.connect(database_file_name)
    cursor = conn.cursor()
    command = (
        f'CREATE TABLE {table_name} ('
        + ', '.join(
            [k + ' ' + PYTHON_TYPE_TO_SQL_TYPE[v] for k, v in columns.items()])
        + ')'
    )
    try:
        cursor.execute(command)
    except sqlite3.OperationalError:
        warnings.warn(f"The table {table_name} already exists")


def initialize_databases(config, config_param_values):
    valid_column_types = {
        "config_idx": int,
        "repeat_idx": int,
        "step": int,
        "learning_rate": float,
        "loss": float
    }
    basic_columns = list(valid_column_types.keys())

    metric_names = sorted(config["train"]["metrics"].keys())
    invalid_metric_names = list(
        set(metric_names) & set(valid_column_types.keys()))
    if invalid_metric_names:
        raise ValueError(
            f'Metric name cannot be one of {list(valid_column_types)}. '
            f'Invalid metric names: {invalid_metric_names}')
    valid_column_types.update({m: float for m in metric_names})

    train_column_types = copy.copy(valid_column_types)

    hook_names = sorted(config["train"]["valid"]["hooks"].keys())
    invalid_hook_names = list(set(hook_names) & set(valid_column_types.keys()))
    if invalid_hook_names:
        raise ValueError(
            f'Hook name cannot be equal to one of metrics or one of elements '
            f'of {basic_columns}. Invalid hook names: {invalid_hook_names}'
        )
    valid_column_types.update({h: float for h in hook_names})

    varied_param_names = sorted(config_param_values.keys())
    invalid_varied_param_names = list(
        set(varied_param_names) & valid_column_types.keys())
    if invalid_varied_param_names:
        raise ValueError(
            f'Varied param name cannot be equal to one of elements of '
            f'{basic_columns}, or to one of metric names, or one of hook '
            f'names. Invalid varied param names: {invalid_varied_param_names}'
        )
    valid_column_types.update(
        {name: type(config_param_values[name]) for name in varied_param_names})

    train_column_types.update(
        {name: type(value) for name, value
         in sorted(config_param_values.items())})

    create_table(
        config["train"]["result_save_path"], 'valid', valid_column_types)

    create_table(
        config["train"]["result_save_path"], 'train', train_column_types)


def main():
    with open(args.config) as f:
        config = json.load(f)
    if args.clear:
        clear(config)
    set_save_paths(config)
    configs, param_values_by_config = expand_param_variations(config)
    initialize_databases(configs[0], param_values_by_config[0])
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
