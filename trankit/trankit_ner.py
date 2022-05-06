import trankit
from trankit.iterators.ner_iterators import NERDataset
import pickle


def save_trainer(trainer, save_path):
	with open(save_path, 'wb') as f:
		pickle.dump(trainer, f, pickle.HIGHEST_PROTOCOL)


def eval_trainer(trainer_path, test_dataset):
	with open(trainer_path, 'rb') as f:
		trainer = pickle.load(f)

	test_set = NERDataset(
		config=trainer._config,
		bio_fpath=test_dataset,
		evaluate=True
	)
	test_set.numberize()
	test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
	result = trainer._eval_ner(
			data_set=test_set,
			batch_num=test_batch_num,
			name='test',
			epoch=-1
	)
	print(result)


def train_test():

	trainer = trankit.TPipeline(
		training_config={
			'max_epoch': 10,
			'category': 'customized-ner',
			'task': 'ner', # task name
			'save_dir': './output', # directory to save the trained model
			'train_bio_fpath': 'data/train.txt',
			'dev_bio_fpath': 'data/dev.txt',
			'max_input_length': 1000,
			'batch_size': 32,
		}
	)

	trainer._config.linear_dropout = 0.3
	trainer._config._cache_dir = './cache'

	print(', '.join("%s: %s" % item for item in vars(trainer._config).items()))

	trainer.train()

	test_set = NERDataset(
		config=trainer._config,
		bio_fpath='data/test.txt',
		evaluate=True
	)
	test_set.numberize()
	test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
	result = trainer._eval_ner(
			data_set=test_set,
			batch_num=test_batch_num,
			name='test',
			epoch=-1
		)
	print(result)
	#save_trainer(trainer, './output_dir/trainer.pkl')

train_test()