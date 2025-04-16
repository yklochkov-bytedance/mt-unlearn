import os
import yaml
import json


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


class Tee:

    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def print(self, message):
        self.write(message + "\n")

    def flush(self):
        self.stream.flush()
        self.file.flush()


def add_mount_path_if_required(path):
    return path.replace("MOUNT_PATH", os.environ['MOUNT_PATH'])


class Logger:
    def __init__(self, path, create_dir=True):
        self.path = path
        # check if path exists and create if not
        if create_dir:
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
        # create file
        with open(self.path, "w") as _:
            pass

    def dump(self, json_serializable_obj):
        with open(self.path, "a") as file:
            file.write(json.dumps(json_serializable_obj) + "\n")
