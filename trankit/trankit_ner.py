import trankit
from trankit.iterators.ner_iterators import NERDataset


def train_test():
	# initialize a trainer for the task
	trainer = trankit.TPipeline(
		training_config={
			'max_epoch': 6,
			'category': 'customized-ner',  # pipeline category
			'task': 'ner', # task name
			'save_dir': './output_dir', # directory to save the trained model
			'train_bio_fpath': './data/train_cleaned.txt', # training data in BIO format
			'dev_bio_fpath': './data/dev_cleaned.txt' # training data in BIO format
		}
	)
	# start training
	trainer.train()

	test_set = NERDataset(
		config=trainer._config,
		bio_fpath='./data/test_cleaned.txt',
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
