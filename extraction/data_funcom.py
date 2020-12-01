import copy
import re

import wget
import tarfile

def get_data(remove_id=True, sample_size=150000):
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
          data.append(sentence)
          size += 1
      dataset.append(data)

  return dataset

def get_characters(remove_id=True, sample_size=150000):
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
          for character in sentence:
            if character not in character_list:
              character_list.append(character)
        
          size += 1

      characters.append(character_list)

  return characters