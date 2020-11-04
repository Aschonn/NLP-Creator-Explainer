import json
from pathlib import Path
import re
import random
import sys

reviews_data = Path("dataset") / "yelp_academic_dataset_review.json"
training_data = Path("filtered_data") / ("fasttext_dataset_training.txt")
test_data = Path("filtered_data") / ("fasttext_dataset_test.txt")

# What percent of data to save separately as test data
percent_test_data = 0.10

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ", string)
    return string

with reviews_data.open() as input, training_data.open("w") as train_output, test_data.open("w") as test_output:

    for line in input:
        review_data = json.loads(line)

        rating = review_data['stars']
        text = review_data['text'].replace("\n", " ")
        text = strip_formatting(text)

        fasttext_line = "__label__{} {}".format(rating, text)

        if random.random() <= percent_test_data:
            test_output.write(fasttext_line + "\n")
        else:
            train_output.write(fasttext_line + "\n")
