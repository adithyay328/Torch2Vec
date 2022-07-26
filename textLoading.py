# Contains utils from pulling text out of the
# IMDB dataset
import random
import os

def cleanReviewText(inText):
  replacements = [("<", ""), ("br", ""), ("/>", ""), ("""\'""", "'")]
  # Removing selected tags
  for rep in replacements:
    inText = inText.replace(rep[0], rep[1])
  
  return inText

# This function allows us to easily grab all text in a given
# review file. This function also handles text cleaning
def getText(baseDir, fName=None, train=None, label=None):
  # If any of the params are none, that means the user
  # wants us to randomly select its value
  if train is None:
    train = random.choice([True, False])
  if label is None:
    label = random.choice([True, False])

  sourceDir = "train"
  if not train:
    sourceDir = "test"
  
  labelDir = "pos"
  if not label:
    labelDir = "neg"

  pathToFiles = f"./{baseDir}/{sourceDir}/{labelDir}"

  if fName is None:
    fileNames = os.listdir(pathToFiles)
    fName = random.choice(fileNames)
  
  with open(f"{pathToFiles}/{fName}", "r") as f:
    res = f.read()
    res = cleanReviewText(res)
    return res