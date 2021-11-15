import torch
import torch.distributed as dist

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def format_step(step):
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += "Epoch: {} ".format(step[0])
    if len(step) > 1:
        s += "Step: {} ".format(step[1])
    if len(step) > 2:
        s += "Validation Step: {} ".format(step[2])
    return s

def format_metric(metric, metadata, value):

    if metadata:
        unit = metadata["unit"] if "unit" in metadata.keys() else ""
        format_str = "{" + metadata["format"] + "}" if "format" in metadata.keys() else "{}"
    else:
        format_table = {
                "average_loss": "{:5.5f}",
                "step_loss": "{:5.5f}",
                "learning_eate": "{:5.5f}"
                       }
        unit = ""
        format_str = format_table[metric] if metric in format_table.keys() else "{}"
    output_str = "{} : {} {}".format(metric, format_str.format(value) if value is not None else value, unit)
    return output_str


