""" An adaptation of the sequence to sequence RNN  by Sean Robertson"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from models_cpu import EncoderRNN, DecoderRNN, AttnDecoderRNN
from tkinter import *




device = torch.device("cpu")

learning_rate=0.01
hidden_size = 256


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 29

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



sql_prefixes = (
    "SELECT", "SELECT *",
    "SELECT Date", "SELECT Issue ",
    "SELECT ZIP code", "SELECT Submitted ",
    "SELECT Complaint ID", "SELECT Sub-issue",
    "SELECT Company", "SELECT Tags ",
    "SELECT Product", "SELECT Timely ",
    "SELECT Consume", "SELECT Sub-product"
    )

def readLangs(language1, language2, reverse=False):

    # Read the file and split into lines
    lines = open('dataset/%s-%s_v3.txt' % (language1, language2), encoding='utf-8').\
    read().strip().split('\n')


    pairs = [[s for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_language = Lang(language2)
        output_language = Lang(language1)
    else:
        input_language = Lang(language1)
        output_language = Lang(language2)

    return input_language, output_language, pairs


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(sql_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(language1, language2, reverse=False):
    input_language, output_language, pairs = readLangs(language2, language1, reverse)
    pairs = filterPairs(pairs)
    for pair in pairs:
        input_language.addSentence(pair[0])
        output_language.addSentence(pair[1])
    return input_language, output_language, pairs

input_language, output_language, _ = prepareData('eng', 'sql', True)


#create an instance of the encoder and the decoder_with_attention for inference
embedding = nn.Embedding(input_language.n_words, hidden_size)
encoder1 = EncoderRNN(input_language.n_words, hidden_size)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_language.n_words, dropout_p=0.1)
encoder_optimizer1 = optim.SGD(encoder1.parameters(), lr=learning_rate)
attn_decoder_optimizer1 = optim.SGD(attn_decoder1.parameters(), lr=learning_rate)




#Load saved-trained model
print("Loading saved models........\n")
checkpoint = torch.load('checkpoints/Model-cp2_v3.pt', map_location=torch.device('cpu'))
embedding_sd = checkpoint['embedding']
encoder1.load_state_dict(checkpoint['encoder'])
attn_decoder1.load_state_dict(checkpoint['decoder'])
encoder_optimizer1.load_state_dict(checkpoint['encoder_optimizer'])
attn_decoder_optimizer1.load_state_dict(checkpoint['decoder_optimizer'])
embedding.load_state_dict(embedding_sd)

# Using gpu
encoder1 = encoder1.to(device)
attn_decoder1 = attn_decoder1.to(device)
embedding =  embedding.to(device)

epoch = checkpoint['epoch']
loss = checkpoint['loss']

print("\n Evaluating models.....")
encoder1.eval()
attn_decoder1.eval()



# Preparing the Data for evaluation
# ---------------------------------
def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, 99) for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


##Evaluation
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_language, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_language.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]








#---------------------------Graphical User Interface----------------------------------
my_interface = Tk()


#list of question prefixes
question_prefix = ['arrange', 'arrange all complaints by product', 'display', 'display all', 'display all complaints',
'display the', 'highlight all complaints', 'how many date received in complaints relating to issue',
'how many Products in complaints relates to tags', 'how many sub product',  'list', 'list all', 'list all items',
'list all complaints', 'list the number of', 'list the number of complaints for CITIBANK', 'list the number of complaints without complaint narrative',
 'number of complaints', 'number of complaints for credit card and credit reporting products', 'picture all complaints', 'show', 'show me ',
 'show me all complaints', 'show me the total number of complaints', 'show the complaints with product identified as credit card',
 'show the product for all complaints with product identified as student loan, debt collection and credit reporting',
 'sort all complaints by', 'what is the total number of complaints']

class AutosuggestionEntry(Entry):
    def __init__(self, question_prefix, *args, **kwargs):

        Entry.__init__(self, *args, **kwargs)
        self.question_prefix = question_prefix
        self.variable = self["textvariable"]
        if self.variable == '':
            self.variable = self["textvariable"] = StringVar()

        self.variable.trace('w', self.changed)
        self.bind("<Right>", self.selection)
        self.bind("<Up>", self.up)
        self.bind("<Down>", self.down)

        self.list_frame_up = False

    def changed(self, name, index, mode):

        if self.variable.get() == '':
            self.list_frame.destroy()
            self.list_frame_up = False
        else:
            words = self.comparison()
            if words:
                if not self.list_frame_up:
                    self.list_frame = Listbox()
                    self.list_frame.bind("<Button-1>", self.selection)
                    self.list_frame.bind("<Right>", self.selection)
                    self.list_frame.config(font=("Courier", 12), width = 60)
                    self.list_frame.place(x=100, y=220)
                    self.list_frame_up = True

                self.list_frame.delete(0, END)
                for w in words:
                    self.list_frame.insert(END,w)
            else:
                if self.list_frame_up:
                    self.list_frame.destroy()
                    self.list_frame_up = False

    def selection(self, event):

        if self.list_frame_up:
            self.variable.set(self.list_frame.get(ACTIVE))
            self.list_frame.destroy()
            self.list_frame_up = False
            self.icursor(END)

    def up(self, event):

        if self.list_frame_up:
            if self.list_frame.curselection() == ():
                index = '0'
            else:
                index = self.list_frame.curselection()[0]
            if index != '0':
                self.list_frame.selection_clear(first=index)
                index = str(int(index)-1)
                self.list_frame.selection_set(first=index)
                self.list_frame.activate(index)

    def down(self, event):

        if self.list_frame_up:
            if self.list_frame.curselection() == ():
                index = '0'
            else:
                index = self.list_frame.curselection()[0]
            if index != END:
                self.list_frame.selection_clear(first=index)
                index = str(int(index)+1)
                self.list_frame.selection_set(first=index)
                self.list_frame.activate(index)

    def comparison(self):
        pattern = re.compile('.*' + self.variable.get() + '.*')
        return [w for w in self.question_prefix if re.match(pattern, w)]


def evaluateQuery():
    input_sentence = question.get()
    try:
        output_words, attentions = evaluate(encoder1, attn_decoder1, input_sentence)
        str = ' '.join(output_words)
        
        #strip off the <EOS> tag
        str_strip = str[:-7]
        sql.insert(END, str_strip + '\n')
        print(sql.get(1.0, END))

    except RuntimeError:
        sql.insert(END, 'Error!, Please rephrase your question' + '\n')
        print('RuntimeError')
    except indexError:
        sql.insert(END, 'Error!, question is too long' + '\n')
        print('Index Error, question is too long')

width = 1000
height = 1000
sz = 30
my_interface.geometry("{}x{}".format(width, height))
my_interface.title("Consumer_complaints Question Answering Interface")
canvas = Canvas(my_interface, width=width, height=height, bg='#ffcccc')
canvas.pack(fill="both", expand=True)
color='white'



titlebar_text = Label(my_interface, text = "Consumer complaints Question Answering Interface", justify = LEFT, bg = 'white', fg='green')
titlebar_text.config(font=("Times New Roman", 35), width = 45)
titlebar_text.place(x = width/10, y=50)


question_label = Label(my_interface, text="QUERY: ", justify = LEFT, bg='#ffcccc', fg='green')
question_label.place(x=width/35, y=225)
question_label.config(font=("Courier", 15))

question = AutosuggestionEntry(question_prefix, my_interface)
question.config(font=("Courier", 15), width = 60)
question.place(x=width/10, y=220)




submit_button = Button(my_interface, text = "Submit", command =evaluateQuery, highlightbackground = '#1a1f71', fg="red")
submit_button.config(font=("Courier", 15), height=1, width = 15)
submit_button.place(x = 850, y= 217)



sql_label = Label(my_interface, text="SQL: ", justify = LEFT, bg='#ffcccc', fg='green')
sql_label.place(x=width/35, y=345)
sql_label.config(font=("Courier", 20))

sql= Text(my_interface)
sql.config(font=("Courier", 15), width = 60)
sql.place(x=width/10, y=340)




exit = Button(my_interface, text = "Exit", command = my_interface.destroy, highlightbackground = '#1a1f71', fg="red")
exit.place(x = 480, y= 950)





my_interface.mainloop()
