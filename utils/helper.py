import yaml

def get_path(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

configs=get_path()