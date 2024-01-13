# Natural Language Interface for Database (NLIDB) using Machine/Deep Learning

This project showcases a Natural Language Interface to a database system, utilizing machine and deep learning techniques. The system interprets English language text instructions, translates them into SQL queries, and retrieves the corresponding data from the database. The primary goal is to demonstrate the effectiveness of neural networks in generating SQL queries through a user-friendly interface.

## Project Overview

The neural network powering this NLIDB was trained on samples derived from an open-source consumer complaints database. The architecture involves:

- **Dataset:** Contains a set of 473 pairs of SQL queries and their corresponding natural language instructions used for training.
- **Checkpoints:** This directory stores the trained model checkpoints.
- **Models:** Implements a sequence-to-sequence RNN
- **Trainer:** Handles the training process using the 473 training examples.
- **Graphical User Interface (GUI):** Utilizes the Python `tkinter` module to accept natural language questions from users and displays the equivalent SQL queries.

## Requirements

Ensure you have the required dependencies installed using:

```bash
pip install -r requirements.txt
```


## Run

Execute the following command to run the graphical user interface:

```bash
python3 GUI_cpu_473_samples.py
```

## How to Use the Application
Type a Natural language (Only English is supported) query or select from a list in the drop down

![Natural language query](https://github.com/algorithm707/SeunAdekoya-git/blob/master/Translation_NaturalLang_to_sql/doc/gui_user_2.png)


Generated SQL 
![Generated SQL for DB query](https://github.com/algorithm707/SeunAdekoya-git/blob/master/Translation_NaturalLang_to_sql/doc/gui_user_3.png)


## Acknowledgments
This project was part of a Software Research Project submitted by Seun Adekoya to the Department of Computer Science at Indiana State University, Terre Haute, Indiana, USA. It was submitted as part of the requirements to obtain a Master's degree in 2018.

Additionally, the learning algorithm employed in this project was adapted from Sean Robertson's encoder-decoder RNN.


Feel free to explore, contribute, and provide feedback. Happy querying! 🚀

