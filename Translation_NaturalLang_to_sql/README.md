# Application of Deep Learning techniques to Natural language interface for database part1
This project demonstrates a simple Natural Language interface to database system which takes in text instruction in English language, translates it to sql queries and returns the retrieved data from database.The aim of this project is to demonstrate the application of neural networks to
generates sql queries from userinput natural language (English) interface for a database - NLIDB.

The network was trained on samples generated from an open source consumer complaints database

The general arrangement includes
dataset: this contain the training example set of 473 pairs of sql and the equivalent natural language
checkpoints: the training model is saved here
models: this is a sequence to sequence RNN adapted from the Sean Robertson's encoder-decoder example
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
trainer: trains the RNN on the 473 training examples
Graphical user interface: uses the python tkinter module to accepts user natural language questions and display
the equivalent sql query to be executed against a database



## Requirements

> pip install -r requirements.txt


## Run

> Python3 GUI_cpu_473_samples.py

