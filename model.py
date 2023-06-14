import random

import pandas as pd
# from huggingface_hub import notebook_login # TODO: why do we need this?
# notebook_login()

# train test function
from sklearn.model_selection import train_test_split
# function to turn csv into dataset
from datasets import load_dataset
import evaluate  # for rouge = evaluate.load("rouge")
# tokenizer
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

"""
https://huggingface.co/docs/transformers/tasks/summarization
"""

# define split sizes
TRAIN_SIZE = 0.7
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.1

# prefix to add to each sample - the instruction for the model
PREFIX = "summarize: "
PRETRAINED_MODEL_NAME = "facebook/bart-large-cnn"


class Main:
    def __init__(self, dataset_path: str, model_name=PRETRAINED_MODEL_NAME):
        # read path to csv that contains dataset into pd
        self.dataset = load_dataset('csv', data_files=[dataset_path])

        # load tokenizer for specific model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # preprocess the dataset
        self.dataset = self.dataset.map(self.__preprocess_function, batched=True)

        # split into train, validation and test
        self.train_set, self.validation_set, self.test_set = self.__split_train_val_test()

        # load all classes for model and trainer
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.training_args = Seq2SeqTrainingArguments(output_dir="my_awesome_billsum_model",
                                                      evaluation_strategy="epoch"
                                                      # TODO: decide specific
                                                      )
        self.trainer = Seq2SeqTrainer(model=self.model, args=self.training_args,
                                      train_dataset=self.train_set,
                                      eval_dataset=self.validation_set,
                                      tokenizer=self.tokenizer,
                                      data_collator=self.data_collator,
                                      compute_metrics=compute_metrics # TODO: define
                                      )



    def __load_dataset(self):
        pass

    def __split_train_val_test(self):
        # split dataset into validation and train, and test
        train_and_validation, test = train_test_split(self.dataset,
                                                      test_size=TEST_SIZE,
                                                      random_state=1)
        train, validation = train_test_split(train_and_validation,
                                             test_size=(VALIDATION_SIZE /
                                                        (VALIDATION_SIZE +
                                                         TRAIN_SIZE)),
                                             random_state=1)
        return train, validation, test

    def __preprocess_function(self, data):
        inputs = [PREFIX + doc for doc in data["text"]]
        model_inputs = self.tokenizer(inputs, max_length=1024, truncation=True)

        labels = self.tokenizer(text_target=data["summary"], max_length=128,
                                truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    model = Main(dataset_path="data.csv")
