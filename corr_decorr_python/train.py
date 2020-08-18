import argparse
import copy
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import warnings
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import torch
from torch.optim import lr_scheduler


PYTHON_TYPE_TO_SQL_TYPE = {
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

def get_args():
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
    return args


def get_nested_dict_elem(d, path):
    for k in path:
        d = d[k]
    return d


def set_nested_dict_elem(d, path, value):
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value


def find_varied_params(config):
    varied_params = {}
    _find_varied_params(varied_params, config, ())
    for param_name, varied_specs in varied_params.items():
        if 'keys' not in varied_specs:
            assert len(varied_specs) > 0
            if len(varied_specs) == 1:
                varied_specs['keys'] = list(varied_specs.values())[0]
            else:
                raise ValueError(
                    f'No keys for parameter "{param_name}". You have to '
                    f'specify "keys" if too fields are varied simultaneously. '
                    f'varied_specs: {varied_specs}')
    return varied_params


def _find_varied_params(params, config, path):
    sub_dict = get_nested_dict_elem(config, path)
    if "type" in sub_dict and sub_dict["type"] == "varied":
        name = sub_dict["name"] if "name" in sub_dict else path[-1]
        if name not in params:
            params[name] = {}
        if "keys" in sub_dict and "keys" in params:
            raise ValueError(
                f'There are 2 sets of keys for parameter "{name}". You '
                f'probably provided excess set of keys or specified the name '
                f'of parameter wrongly. params[name]={params[name]}. '
                f'subdict={sub_dict}')
        if "keys" in sub_dict:
            params[name]["keys"] = sub_dict["keys"]
        params[name][path] = sub_dict["values"]
    else:
        for k, v in sub_dict.items():
            if isinstance(v, dict):
                _find_varied_params(params, config, path + (k,))


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
    check_path_is_not_varied(config["train"], "result_save_path")
    check_path_is_not_varied(config["train"], "model_save_path")
    configs, param_values_by_config = [config], [{}]
    varied_params = find_varied_params(config)
    for param_name, varied_specs in varied_params.items():
        paths_and_values = [
            (p, v) for p, v in varied_specs.items() if isinstance(p, tuple)]
        paths, values = zip(*paths_and_values)
        new_configs, new_param_values_by_config = [], []
        for c, c_p in zip(configs, param_values_by_config):
            for set_idx, one_values_set in enumerate(zip(*values)):
                new_c = copy.deepcopy(c)
                new_c_p = copy.deepcopy(c_p)
                new_c_p[param_name] = varied_specs["keys"][set_idx]
                for i, v in enumerate(one_values_set):
                    set_nested_dict_elem(new_c, paths[i], v)
                new_configs.append(new_c)
                new_param_values_by_config.append(new_c_p)
        configs = new_configs
        param_values_by_config = new_param_values_by_config
    return configs, param_values_by_config


def get_vars(config):
    if "metadata" in config and "vars" in config["metadata"]:
        vars_ = config["metadata"].get("vars")
    else:
        vars_ = {}
    return vars_


def expand_vars_in_config(config, config_vars):
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, str):
                config[k] = v.format(**config_vars)
            else:
                expand_vars_in_config(v, config_vars)
    elif isinstance(config, list):
        for i, v in enumerate(config):
            if isinstance(v, str):
                config[i] = v.format(**config_vars)
            else:
                expand_vars_in_config(v, config_vars)


def set_save_paths(config):
    config_vars = get_vars(config)
    root_path = config_vars.get(
        "ROOT", "~/neuron-correlation-decorrelation-paper")
    try:
        config_rel_path = \
            Path(__file__).relative_to(
                Path(root_path).expanduser() / Path('code/configs'))
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
            Path(root_path) / Path("results") \
            / config_rel_path.with_suffix(".db")
    if "model_save_path" not in config["train"]:
        config["train"]["model_save_path"] = \
            Path(root_path) / Path("saved_models") / config_rel_path


def import_object(obj_import_path):
    elems = obj_import_path.split('.')
    module_name = '.'.join(elems[:-1])
    old_sys_path_0 = sys.path[0]
    sys.path[0] = str(Path(__file__).parent)
    module = importlib.import_module(module_name)
    sys.path[0] = old_sys_path_0
    return getattr(module, elems[-1])


def instantiate_object_from_config(config, args=()):
    config = copy.deepcopy(config)
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(*args, **config)


def build_model(config):
    cls_ = import_object(config["class"])
    del config["class"]
    return cls_(**config)


class Stopper:
    def __init__(self, train_config):
        self.train_config = train_config
        self.stop_patience = train_config.get("stop_patience")
        self.num_steps = train_config.get("num_steps")
        self.valid_period = train_config["valid_period"]
        self.method_of_stopping = self.get_training_interruption_method()

        self.best_v_loss = float('inf')
        self.stop_impatience = 0

    def state_dict(self):
        return {"best_v_loss": self.best_v_loss,
                "stop_impatience": self.stop_impatience}

    def load_state_dict(self, sd):
        self.best_v_loss = sd['best_v_loss']
        self.stop_impatience = sd['stop_impatience']

    def get_training_interruption_method(self):
        if 'num_steps' in self.train_config:
            method_of_interrupting_training = 'fixed_num_steps'
        elif "num_epochs" in self.train_config:
            method_of_interrupting_training = 'fixed_num_epochs'
        elif 'stop_patience' in self.train_config:
            method_of_interrupting_training = 'impatience'
        else:
            raise ValueError(
                "Train config has to contain either parameter 'num_steps', or "
                "parameter 'num_epochs', or parameter 'stop_patience'.\n"
                "train config={}".format(self.train_config)
            )
        return method_of_interrupting_training

    def step(self, step, epoch, v_loss):
        if self.method_of_stopping == 'fixed_num_steps':
            stop_training = step > self.num_steps
        elif self.method_of_stopping == 'impatience':
            if step % self.valid_period == 0:
                if v_loss < self.best_v_loss:
                    self.stop_impatience = 0
                    self.best_v_loss = v_loss
                else:
                    self.stop_impatience += 1
            stop_training = self.stop_impatience > self.stop_patience
        else:
            raise ValueError(
                "Unsupported method of interrupting training '{}'".format(
                    self.method_of_stopping
                )
            )
        return stop_training


def time_for_logarithmic_logging(step, factor):
    if step == 0:
        return True
    step_is_integer_power_of_factor = int(np.log(step+1) / np.log(factor)) \
        - int(np.log(step) / np.log(factor)) > 0
    return step_is_integer_power_of_factor


def compute_metrics(pred_log_probas, labels, metric_specs):
    metric_values = {}
    for m, func_import_path in metric_specs.items():
        metric_values[m] = import_object(func_import_path)(
            labels, pred_log_probas)
    return metric_values


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
    database_path = Path(database_path).expanduser()
    conn = sqlite3.connect(database_path)
    cursor = conn.execute(f"SELECT * FROM {table_name}")
    col_names = [description[0] for description in cursor.description]
    data = {"config_idx": config_idx, "repeat_idx": repeat_idx, "step": step,
            "learning_rate": learning_rate, "loss": loss}
    data.update(metric_values)
    data.update(hooks)
    data.update(varied_params)
    values = []
    for col_name in col_names:
        values.append(data.get(col_name))
    command = f'INSERT INTO {table_name} VALUES (' \
              + ', '.join(['?']*len(values)) \
              + ')'
    cursor.execute(command, tuple(values))
    conn.commit()
    conn.close()


def append_step_metrics(accumulated_metrics, step_metrics):
    for k, v in step_metrics.items():
        accumulated_metrics[k].append(v)


def average_metrics(accumulated_metrics):
    averaged = {}
    for k, v in accumulated_metrics.items():
        averaged[k] = sum(v) / len(v)
    return averaged


def post_process_hooks(model, hook_post_processing):
    return {}  # TODO


def test(
        data_loader,
        model,
        data_type,
        train_config
):
    loss_fn = instantiate_object_from_config(train_config["loss"])
    model.eval()
    required_metrics = train_config.get("metrics", {})
    accumulated_metrics = {k: [] for k in required_metrics}
    accumulated_loss = []
    for test_step, samples in enumerate(data_loader):
        inputs, labels = samples
        pred_log_probas = model(inputs)
        step_metrics = compute_metrics(pred_log_probas, labels, required_metrics)
        append_step_metrics(accumulated_metrics, step_metrics)
        accumulated_loss.append(
            loss_fn(
                pred_log_probas.reshape((-1,) + pred_log_probas.shape[-1:]),
                labels.reshape([-1])
            ).detach().numpy()
        )
    metrics = average_metrics(accumulated_metrics)
    accumulated_loss = sum(accumulated_loss) / len(accumulated_loss)
    hook_values = post_process_hooks(
        model, train_config[data_type]["hooks_post_processing_fns"])
    return metrics, accumulated_loss, hook_values


def log(config_idx, repeat_idx, step, data_type, loss, metric_values, lr):
    print(f'config: {config_idx}, repeat: {repeat_idx}, step: {step}, "data": '
          f'{data_type}, "loss": {loss}, "metrics": {metric_values}, '
          f'"learning rate": {lr}')


def get_data_loaders(config):
    datasets = {}
    if "vocab" in config:
        ds_args = (instantiate_object_from_config(config["vocab"]),)
    else:
        ds_args = ()
    if "train_dataset" in config:
        datasets['train'] = instantiate_object_from_config(
            config["train_dataset"], ds_args)
    if "valid_dataset" in config:
        datasets['valid'] = instantiate_object_from_config(
            config["valid_dataset"], ds_args)
    if "test_dataset" in config:
        datasets['test'] = instantiate_object_from_config(
            config["test_dataset"], ds_args)
    data_loaders = {}
    if "train_data_loader" in config:
        data_loaders['train'] = instantiate_object_from_config(
            config["train_data_loader"], (datasets['train'],))
    if "valid_data_loader" in config:
        data_loaders['valid'] = instantiate_object_from_config(
            config["valid_data_loader"], (datasets['valid'],))
    if "test_data_loader" in config:
        data_loaders['test'] = instantiate_object_from_config(
            config['test_data_loader'], (datasets['test'],))
    return data_loaders


def get_element_params(data_config, data_loaders):
    if "vocab" in data_config:
        return len(data_loaders["train"].dataset.vocab)
    else:
        pass  # TODO


def remove_first_dim(inputs, labels):
    if inputs.shape[0] > 1 or labels.shape[0] > 1:
        raise ValueError(
            f'If "remove_first_dim" value is `true` the first '
            f'dimension of `inputs` and `labels` has to be 1. '
            f'inputs.shape={inputs.shape} '
            f'labels.shape={labels.shape}')
    inputs = inputs.reshape(inputs.shape[1:])
    labels = labels.reshape(labels.shape[1:])
    return inputs, labels


class Logger:
    def __init__(
            self,
            train_config,
            config_idx,
            repeat_idx,
            optimizer,
            valid_data_loader,
            model,
            varied_params):
        self.train_config = copy.deepcopy(train_config)
        self.config_idx = config_idx
        self.repeat_idx = repeat_idx
        self.optimizer = optimizer
        self.valid_data_loader = valid_data_loader
        self.model = model
        self.varied_params = copy.deepcopy(varied_params)
        self.log_factor = self.train_config["log_factor"]
        self.save_path = self.train_config["result_save_path"]

    def log_and_save_if_it_is_time(self, step, train_loss, train_metrics):
        if time_for_logarithmic_logging(step, self.log_factor):
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.detach().numpy()
            lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
            v_metrics, v_loss, v_hooks = test(
                self.valid_data_loader, self.model, "valid", self.train_config)
            save_metrics_hooks_varied(
                self.save_path, 'valid', self.config_idx,
                self.repeat_idx, step, lr,
                v_loss, v_metrics, v_hooks, self.varied_params
            )
            log(self.config_idx, self.repeat_idx, step, 'valid', v_loss,
                v_metrics, lr)
            if step > 0:
                save_metrics_hooks_varied(
                    self.save_path, 'train', self.config_idx,
                    self.repeat_idx, step, lr, train_loss, train_metrics,
                    {}, self.varied_params)
                log(self.config_idx, self.repeat_idx, step, 'train',
                    train_loss, train_metrics, lr)


def train_step(model, loss_fn, optimizer, inputs, labels, train_config):
    model.train()
    optimizer.zero_grad()
    pred_log_probas = model(inputs)
    t_loss = loss_fn(
        pred_log_probas.reshape((-1,) + pred_log_probas.shape[-1:]),
        labels.reshape([-1])
    )
    t_loss.backward()
    optimizer.step()
    t_metrics = compute_metrics(
        pred_log_probas, labels, train_config.get("metrics", {}))
    return t_loss, pred_log_probas, t_metrics


class Scheduler:
    def __init__(self, scheduler_config, optimizer):
        self.scheduler = instantiate_object_from_config(
            scheduler_config, optimizer)

    def step(self, valid_loss, epoch):
        if type(self.scheduler) == lr_scheduler.CosineAnnealingWarmRestarts:
            self.scheduler.step(epoch)
        elif type(self.scheduler) == lr_scheduler.ReduceLROnPlateau:
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

    def state_dict(self):
        return self.scheduler.save_dict()

    def load_state_dict(self, sd):
        self.scheduler.load_save_dict(sd)


def get_checkpoint_path(model_save_path, config_idx, repeat_idx, name):
    return Path(model_save_path).expanduser() / (str(config_idx)) \
           / Path(name + str(repeat_idx) + '.pt')


class Checkpointer:
    def __init__(
            self,
            config_idx,
            repeat_idx,
            train_config,
            model,
            optimizer,
            scheduler,
            stopper
    ):
        self.config_idx = config_idx
        self.repeat_idx = repeat_idx
        self.last_checkpoint_path = get_checkpoint_path(
            train_config["model_save_path"], self.config_idx, self.repeat_idx,
            'name')
        self.best_checkpoint_path = get_checkpoint_path(
            train_config["model_save_path"], self.config_idx, self.repeat_idx,
            'name')
        self.save_path = train_config["model_save_path"]
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.stopper = stopper
        self.best_valid_loss = float('inf')

    def update_checkpoints(self, step, epoch, valid_loss):
        self.create_checkpoint(self.last_checkpoint_path, step, epoch)
        if valid_loss < self.best_valid_loss:
            self.create_checkpoint(self.best_checkpoint_path, step, epoch)
            self.best_valid_loss = valid_loss

    def state_dict(self):
        return {'best_valid_loss': self.best_valid_loss}

    def load_state_dict(self, sd):
        self.best_valid_loss = sd['best_valid_loss']

    def create_checkpoint(self, path, step, epoch):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        backup = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': self.model.parameters(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'stopper_state_dict': self.stopper.state_dict(),
            'checkpointer_state_dict': self.state_dict()
        }
        print(backup)
        torch.save(backup, path)

    def load_checkpoint(self, path):
        backup = torch.load(path)
        self.model.load_state_dict(backup['model'])
        self.optimizer.load_state_dict(backup['optimizer_state_dict'])
        self.scheduler.load_state_dict(backup['scheduler_state_dict'])
        self.stopper.load_state_dict(backup['stopper_state_dict'])
        self.load_state_dict(backup['checkpointer_state_dict'])
        return backup['step'], backup['epoch']

    @staticmethod
    def check_checkpoint_name(name):
        if name not in ['best', 'last']:
            raise ValueError(f'Unsupported checkpoint name "{name}"')

    def load_by_name(self, name):
        self.check_checkpoint_name(name)
        if name == 'best':
            return self.load_checkpoint(self.best_checkpoint_path)
        elif name == 'last':
            return self.load_checkpoint(self.last_checkpoint_path)

    def exists(self, name):
        self.check_checkpoint_name(name)
        if name == 'best':
            return self.best_checkpoint_path.is_file()
        elif name == 'last':
            return self.last_checkpoint_path.is_file()


def train(
        train_config,
        data_loaders,
        model,
        varied_params,
        config_idx,
        repeat_idx):
    optimizer = instantiate_object_from_config(
        train_config["optimizer"], (model.parameters(),))
    loss_fn = instantiate_object_from_config(train_config["loss"])
    scheduler = instantiate_object_from_config(
        train_config["scheduler"], (optimizer,))
    stopper = Stopper(train_config)
    checkpointer = Checkpointer(
        config_idx, repeat_idx, train_config, model, optimizer, scheduler,
        stopper)
    if checkpointer.exists('last'):
        step, epoch = checkpointer.load_by_name('last')
    elif checkpointer.exists('best'):
        step, epoch = checkpointer.load_by_name('best')
    else:
        step, epoch = 0, 0

    logger = Logger(
        train_config, config_idx, repeat_idx, optimizer, data_loaders['valid'],
        model, varied_params)

    train_iterator = enumerate(data_loaders['train'])
    t_loss, t_metrics = None, None
    while True:
        try:
            step, samples = next(train_iterator)
        except StopIteration:
            epoch += 1
            train_iterator = enumerate(data_loaders['train'])
            step, samples = next(train_iterator)
        inputs, labels = samples
        logger.log_and_save_if_it_is_time(step, t_loss, t_metrics)
        if step % train_config["valid_period"] == 0:
            _, v_loss, _ = test(
                data_loaders['valid'], model, "valid", train_config)
            scheduler.step(v_loss, epoch)
            checkpointer.update_checkpoints(step, epoch, v_loss)
            if stopper.step(step, epoch, v_loss):
                break

        t_loss, _, t_metrics = train_step(
            model, loss_fn, optimizer, inputs, labels, train_config)


def get_done_file_path(model_save_path):
    return Path(model_save_path).expanduser() / Path("done.json")


def mark_repeat_as_done(config_idx, repeat_idx, model_save_path):
    done_file_path = get_done_file_path(model_save_path)
    if done_file_path.is_file():
        with done_file_path.open('+') as f:
            done = json.load(f)
            if str(config_idx) not in done:
                done[str(config_idx)] = []
            if repeat_idx in done[str(config_idx)]:
                raise ValueError(
                    f"Repeat {repeat_idx} of config {config_idx} is marked as "
                    f"already made in file {done_file_path}.")
            done[str(config_idx)].append(repeat_idx)
            f.truncate(0)
            json.dump(done, f)


def build_and_run_once(config, varied_params, config_idx, repeat_idx):
    data_loaders = get_data_loaders(config["data"])
    element_params = get_element_params(config["data"], data_loaders)

    model_config = copy.deepcopy(config["model"])
    model_config["element_params"] = element_params
    model = instantiate_object_from_config(model_config)
    train(
        config["train"],
        data_loaders,
        model,
        varied_params,
        config_idx,
        repeat_idx
    )
    get_checkpoint_path(config["train"]["model_save_path"], config_idx,
                        repeat_idx, 'last').unlink()
    get_checkpoint_path(config["train"]["model_save_path"], config_idx,
                        repeat_idx, 'best').unlink()
    mark_repeat_as_done(
        config_idx, repeat_idx, config["train"]["model_save_path"])


def get_remaining_repeats(config_idx, num_repeats, model_save_path):
    done_file = get_done_file_path(model_save_path)
    if done_file.is_file():
        with done_file.open() as f:
            done = json.load(f)
    else:
        done = {}
    return set(range(num_repeats)) - set(done.get(str(config_idx), []))


def build_and_run_repeatedly(config, varied_params, config_idx):
    for repeat_idx in get_remaining_repeats(
            config_idx, config["num_repeats"],
            config["train"]["model_save_path"]):
        build_and_run_once(config, varied_params, config_idx, repeat_idx)


def remove(p):
    if p.is_dir():
        shutil.rmtree(p)
    elif p.is_file():
        p.unlink()
    else:
        raise OSError


def clear(config):
    if "result_save_path" in config["train"]:
        p = Path(config["train"]["result_save_path"]).expanduser()
        remove(p)
    if "model_save_path" in config["train"]:
        p = Path(config["train"]["model_save_path"]).expanduser()
        remove(p)

def create_table(database_file_name, table_name, columns):
    database_file_name = Path(database_file_name).expanduser()
    database_file_name.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(database_file_name))
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
        warnings.warn(f'The table "{table_name}" already exists in database '
                      f'"{database_file_name}"')


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


def get_all_content(path):
    content = []
    for root, dirs, files in os.walk(path):
        root = Path(root).relative_to(path)
        for dir_ in dirs:
            content.append(root / Path(dir_))
        for fn in files:
            content.append(root / Path(fn))
    return content


def download(download_config, paths):
    data_path = Path(download_config["data_path"].format(**paths)).expanduser()
    file_name = Path(download_config["url"].split('/')[-1])
    downloaded_archive = Path(
        "{ROOT}/{archive_dir}".format(
            **paths, archive_dir="cached_download")) / file_name
    if not downloaded_archive.is_file():
        downloaded_archive.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(download_config["url"], downloaded_archive)
    if str(downloaded_archive).endswith('.tar.gz'):
        with tarfile.open(downloaded_archive, 'r:gz') as tarf:
            if set(Path(i) for i in tarf.getnames()) \
                    != set(get_all_content(data_path)):
                tarf.extractall(data_path)
    if str(downloaded_archive).endswith('.zip'):
        with ZipFile(downloaded_archive, 'r') as zipf:
            if set(Path(i) for i in zipf.namelist()) \
                    != set(get_all_content(data_path)):
                zipf.extractall(data_path)


def main():
    args = get_args()
    with open(args.config) as f:
        config = json.load(f)
    set_save_paths(config)
    config_vars = get_vars(config)
    expand_vars_in_config(config, config_vars)
    if args.clear:
        clear(config)
    configs, param_values_by_config = expand_param_variations(config)
    initialize_databases(configs[0], param_values_by_config[0])
    for i, (varied_params, config) in enumerate(
            zip(param_values_by_config, configs)):
        download_configs = config.get("metadata", {}).get("downloads", [])
        for c in download_configs:
            download(c, get_vars(config))
        param_set_path = \
            Path(config["train"]["model_save_path"]).expanduser().parent \
            / Path(f'param_sets/{i}.json')
        param_set_path.parent.mkdir(parents=True, exist_ok=True)
        with param_set_path.open('w') as f:
            json.dump(varied_params, f)
        build_and_run_repeatedly(config, varied_params, i)


if __name__ == '__main__':
    main()
