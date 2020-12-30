import logging
import logging.config
import yaml


def load_config(filepath):
	with open(filepath) as file:
		params = yaml.safe_load(file)

	return params 


def set_logging(filepath):
	logging.config.fileConfig(filepath)
	log = logging.getLogger("default")

	return log