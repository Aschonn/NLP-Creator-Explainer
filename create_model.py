import json
import re
import fasttext
from pathlib import Path

nameofmodel = 'example'

training_data = "filtered_data/fasttext_dataset_training.txt"
test_data = "filtered_data/fasttext_dataset_test.txt"

model = fasttext.train_supervised(training_data)

model.test(test_data)

model.save_model(f"models/{nameofmodel}.bin")