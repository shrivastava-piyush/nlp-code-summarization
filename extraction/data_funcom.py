import copy
import re

import pandas as pd

import wget
import tarfile

def get_data(remove_id=True, sample_size=20000, start_token='\t', end_token='\n'):
  dataset = []
  dataset_name = wget.download("https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_tokenized.tar.gz")
  tar = tarfile.open("funcom_tokenized.tar.gz", "r:gz")
  #print([member.name for member in tar.getmembers()])

  for member in tar.getmembers():
    data = []
    size = 0
    if member.name == 'funcom_tokenized/comments' or member.name == 'funcom_tokenized/functions':
      file = tar.extractfile(member)
      for line in file:
        if size < sample_size:
          sentence = copy.copy(line.decode())
          sentence = re.sub(r'^.*?\t', '', sentence)
          sentence = re.sub(r'\n', '', sentence)
          if member.name == 'funcom_tokenized/comments':
            sentence = start_token + sentence + end_token
          data.append(sentence)
          size += 1
      dataset.append(data)

  return dataset

def get_characters(remove_id=True, sample_size=20000):
  characters = []
  dataset_name = wget.download("https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_tokenized.tar.gz")
  tar = tarfile.open("funcom_tokenized.tar.gz", "r:gz")

  for member in tar.getmembers():
    character_list = []
    size = 0
    if member.name == 'funcom_tokenized/comments' or member.name == 'funcom_tokenized/functions':
      file = tar.extractfile(member)
      for line in file:
        if size < sample_size:
          sentence = copy.copy(line.decode())
          sentence = re.sub(r'^.*?\t', '', sentence)
          sentence = re.sub(r'\n', '', sentence)
          if member.name == 'funcom_tokenized/comments':
            sentence = '\t' + sentence + '\n'
          for character in sentence:
            if character not in character_list:
              character_list.append(character)
        
          size += 1

      character_list = sorted(character_list)
      characters.append(character_list)

  return characters

def get_json_data(remove_id=True, sample_size=20000, start_token='\t', end_token='\n'):
  dataset = []
  dataset_name = wget.download("https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_filtered.tar.gz")
  tar = tarfile.open("funcom_filtered.tar.gz", "r:gz")

  for member in tar.getmembers():
    data = []
    size = 0
    if member.name == 'funcom_processed/comments.json' or member.name == 'funcom_processed/functions.json':
      file = tar.extractfile(member)

      data_df = pd.read_json(file.read(), orient='index')
      data_df.reset_index(drop=True, inplace=True)
      data_df.columns = ['Column']

      if member.name == 'funcom_filtered/comments.json':
          data_df['Column'] = start_token + data_df['Column'] + end_token
      size += 1
      dataset.append(data_df['Column'][:sample_size].tolist())

  return dataset