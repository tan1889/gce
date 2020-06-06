import os
import configparser
config = configparser.ConfigParser()
config.sections()
config.read(os.path.dirname(__file__) + '/greedy.config')

for key in ['datasets', 'checkpoints']:
	config['DEFAULT'][key] = os.path.expanduser(config['DEFAULT'][key])
