import os

import yaml


def read_yaml(config):
    if os.path.exists(config):
        with open(config, 'r') as f:
            return yaml.load(f)
    else:
        raise Exception("Config file %s does not exist" % config)
