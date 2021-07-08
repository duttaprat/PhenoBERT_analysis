import torch
import torch.nn as nn
import os
import pickle
#import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from plotter import loss_curve_plot
from collections import OrderedDict
import datetime
import time
import math
from evaluation_functions import *
from PhenoBERT_functions import *



os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys


ThresholdScore= float(sys.argv[1])
MaxCandidatePhenotype = int(sys.argv[2])



print ("Reading the datasets.....")
t0 = time.time()
data = pd.read_csv('../Phenotypic_data/phenotype_original.csv')
phenotype_data = pd.read_csv("../Phenotypic_data/phenotypic_abnormality_descendants_synonyms.csv")
print (time.time()-t0)


phenotype_data = phenotype_data.head(1024)


graph= None
t = time.time()
print ("\nHPO Graph is creating")


# t1= time.time()
# graph = create_graph(data)
# print (time.time() - t1)
# print ("HPO Graph is created.\n")


t2= time.time()
graph = pickle.load(open("phenotype_graph.pickle", "rb"))
print (time.time() - t2)


# ## Parameters

bert_model = "allenai/scibert_scivocab_uncased"  # 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2', 'bert-base-uncased', ...
freeze_bert = True  # if True, freeze the encoder weights and only update the classification layer weights
maxlen = 128  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
bs = 256 # batch size
iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = 1e-3  # learning rate
epochs = 3000  # number of training epochs




print("Creation of the results' folder...")
result_directory = "../results"
if not os.path.exists(result_directory):
    os.mkdir(result_directory)
else:
    print("Folder already exists.")


# In[ ]:



path_to_model = '../our_fine_tuned_model.pt'
# path_to_model = '/content/models/...'  # You can add here your trained model

path_to_output_file = '../results/output_validation_check2.txt'
#path_to_output_file2 = 'results/output_train.txt'



print ("Fine-tuned model is loading.\n")
model = SentencePairClassifier(bert_model)

print("Loading the weights of the model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:  # if multiple GPUs
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    new_state_dict = OrderedDict()
    state_dict = torch.load(path_to_model)
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    model.load_state_dict(new_state_dict)

print()


# Uncomment the following line if you want to use only one GPU.
model.load_state_dict(torch.load(path_to_model))
model.to(device)
a1 = time.time()
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
print("Quantization time: {}".format(time.time() - a1))
print ("Fine-tuned model is loaded..\n")


print ("Which type of input you want to give to the model. Please choose from the following.")
print ("=====================================================================================")
print ("Single sentence: Press 1")
print ("Multiple sentences: Press 2")
print ("Please enter your choice:")
choice = input()

if (int(choice) ==1):
	print ("Please enter the sentence:")
	s = input() 
	df = create_dataset(s, phenotype_data)
	#print(df.head(100))

	print("Reading test data...")
	test_set = CustomDataset(df, maxlen, bert_model)
	test_loader = DataLoader(test_set, batch_size=bs, num_workers=1)

	print("Predicting on test data...")
	t3= time.time()
	test_prediction(net=model, device=device, dataloader=test_loader, df=df, with_labels=True, result_file=path_to_output_file)
	
	print("Prediction time:", time.time()-t3)
	print("Predictions are available in : {}".format(path_to_output_file))
	#print (df)

	df_sort_new = allSortedHPO(df, phenotype_data)
	df_sort_new.to_csv('../results/PhenoBERT_score_synonym.csv',  index = False)

	df_new = TopHPOs(df_sort_new, ThresholdScore, MaxCandidatePhenotype)
	print ("###\n")

	#print (df_new.head(30))
	if (int(sys.argv[3])==1):
		#print ("Please enter the list of phenotypes obtained from DOC2HPO:")
		Doc2HPO_list = ['HP:0002119', 'HP:0007099', 'HP:0000076', 'HP:0001273', 'HP:0000238', 'HP:0000079', 'HP:0008443'] 
		#Doc2HPO_list = input()
		Doc2HPOIndependentPhenotypes(df_new, Doc2HPO_list, graph, df_sort_new)
	elif (int(sys.argv[3])==0):
		print ("The independent phenotypes are:")
		IndependentPhenotypes(df_new, graph)

elif (int(choice) ==2):
	print ("Put all the multiple sentences in a csv file and enter the name of csv file:")
	file = input()
	all_sentence_data = pd.read_csv(file)
	temp= 0
	for sentence in all_sentence_data['Sentence']:
		print ("\nSentence ", temp,": ", sentence)
		s= sentence
		df = create_dataset(s, phenotype_data)
		#print(df.head(100))

		print("Reading test data...")
		test_set = CustomDataset(df, maxlen, bert_model)
		test_loader = DataLoader(test_set, batch_size=bs, num_workers=1)

		print("Predicting on test data...")
		test_prediction(net=model, device=device, dataloader=test_loader, df=df, with_labels=True, result_file=path_to_output_file)
		print()
		print("Predictions are available in : {}".format(path_to_output_file))
		#print (df)

		df_sort_new = allSortedHPO(df, phenotype_data)
		df_sort_new.to_csv('results/PhenoBERT_score_synonym_'+str(temp)+'.csv',  index = False)
		temp= temp +1
		df_new = TopHPOs(df_sort_new, ThresholdScore, MaxCandidatePhenotype)
		print ("###\n")


		#print (df_new.head(30))
		if (int(sys.argv[3])==1):
			#print ("Please enter the list of phenotypes obtained from DOC2HPO:")
			Doc2HPO_list = ['HP:0002119', 'HP:0007099', 'HP:0000076', 'HP:0001273', 'HP:0000238', 'HP:0000079', 'HP:0008443'] 
			#Doc2HPO_list = input()
			Doc2HPOIndependentPhenotypes(df_new, Doc2HPO_list, graph, df_sort_new)
		elif (int(sys.argv[3])==0):
			print ("The independent phenotypes are:")
			IndependentPhenotypes(df_new, graph)







#print ("Updated DOC2HPO list", Doc2HPO_list)

print ("\n")


