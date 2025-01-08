import numpy as np
import pandas as pd
import json
from collections import Counter
import ast
from ast import literal_eval as read_dict

def getJsonInfo(filename):
    with open(filename,'r') as f:
        data = json.load(f)
    # Print frequencies
    for key in data.keys():
        print(f"-----{key} (n = {len(data[key])})-----")
        firstCats = [x['labels'][0] for x in data[key]]
        freqs = dict(Counter(firstCats))
        for item in freqs.keys():
            print(f"{item.strip()} -> {freqs[item]}, {freqs[item]/len(data[key])*100:.2f}%")
    
def getInfo(csvfile):
    df = pd.read_csv(csvfile)
    studios = np.unique(df.studio_title)
    for studio in studios:
        total = len(df.loc[df.studio_title==studio])
        print(f"-----{studio} (n = {total})-----")
        data = [df['categorizations'][i] for i in range(len(df)) if df['studio_title'][i] == studio and type(df['categorizations'][i]) is not float]
        firstCats = [ast.literal_eval(y)['labels'][0] for y in data]
        freqs = dict(Counter(firstCats))
        for item in freqs.keys():
            print(f"{item.strip()} -> {freqs[item]}, {freqs[item]/total*100:.2f}%")
        noCats = len([i for i in range(len(df)) if df['studio_title'][i] == studio and type(df['categorizations'][i]) is float])
        print(f"Uncategorized -> {noCats}, {noCats/total*100:.2f}%")

def getUsersInfo(csvfile,keys):
    df = pd.read_csv(csvfile)
    for key in keys:
        total = len(df)
        print(f"-----{key.replace("_"," ").capitalize()} (n = {total})-----")
        data = [df[key+'_categorization'][i] for i in range(len(df)) if type(df[key+'_categorization'][i]) is not float]
        firstCats = [ast.literal_eval(y)['labels'][0] for y in data]
        freqs = dict(Counter(firstCats))
        for item in freqs.keys():
            print(f"{item.strip()} -> {freqs[item]}, {freqs[item]/total*100:.2f}%")
        noCats = len([i for i in range(len(df)) if type(df[key+'_categorization'][i]) is float])
        print(f"Uncategorized -> {noCats}, {noCats/total*100:.2f}%")

# Need to be able to parse the ones with `studio-title` efficiently

def getExamples(csvfile,keys=None,fixDicts=True,threshold=0.9,singleCol=True,specificItem=None):
    # Check if the user only wants to parse a single column with categorizations that correspond to different things (i..e, studio titles)
    if singleCol:
    # if the user only wants a specific column
        if type(keys) is list and len(keys) == 1:
            key = keys[0]
        elif type(keys) is str:
            key = keys
        elif type(keys) is list and len(keys) >= 1:
            raise Exception("You specified `singleCol=True` and specified multiple keys. Please only specify one CSV key for single column example extraction.")
        elif keys is None:
            raise Exception("Please specify a key from which the function can generate examples based on a field in the dataset.")
        
        df = pd.read_csv(csvfile)
        # If the user only wants to retrieve categorizations linked to a specific key in a certain column in the dataframe that is a unique identifier (i.e., studio title)
        if specificItem is not None:
            examples = {}
            if fixDicts:
                cats1 = [read_dict(x) for x in df.loc[df[key]==specificItem]['categorizations'] if type(x) is not float]
            else:
                cats1 = [x for x in df.loc[df[key]==specificItem]['categorizations'] if type(x) is not float]
            categories = cats1[0]['labels']
            for category in categories:
                curList = []
                for i in range(len(cats1)):
                    if cats1[i]['labels'][0] == category and cats1[i]['scores'][0] >= threshold:
                        curList.append(cats1[i]['sequence'])
                examples[category] = curList
            print(f"Examples generated for where {key} is {specificItem}")
            return examples
        else:
            allExamples = {}
            unique_fields = np.unique(df[key]).tolist()
            for field in unique_fields:
                examples = {}
                if fixDicts:
                    cats1 = [read_dict(x) for x in df.loc[df[key]==field]['categorizations'] if type(x) is not float]
                else:
                    cats1 = [x for x in df.loc[df[key]==field]['categorizations']]
                categories = cats1[0]['labels']
                for category in categories:
                    curList = []
                    for i in range(len(cats1)):
                        if cats1[i]['labels'][0] == category and cats1[i]['scores'][0] >= threshold:
                            curList.append(cats1[i]['sequence'])
                    examples[category] = curList
                allExamples[field] = examples
            return allExamples
    elif not singleCol:
        if specificItem:
            raise Exception("You passed a `specificItem` parameter without enabling `singleCol`. Please remove the parameter or ensure this is the correct behavior for your application.")
        if type(keys) is not list:
            keys = [keys]
        elif keys is None:
            raise Exception("Please specify which keys you would like to parse in the CSV.")
        allExamples = {}
        df = pd.read_csv(csvfile)
        for key in keys:
            examples = {}
            if fixDicts:
                curCats = [read_dict(x) for x in df[key] if type(x) is not float]
            else:
                curCats = [x for x in df[key] if type(x) is not float]
            categories = curCats[0]['labels']
            for category in categories:
                curList = []
                for i in range(len(curCats)):
                    if curCats[i]['labels'][0] == category and curCats[i]['scores'][0] >= threshold:
                        curList.append(curCats[i]['sequence'])
                examples[category] = curList
            allExamples[key] = examples
        if len(allExamples.keys()) == 1:
            return allExamples[keys[0]]
        else:
            return allExamples

    

def getCategoryDist(csvfile,keys=None,fixDicts=True,singleCol=True,specificItem=None):
    #! Write documentation for this function and the one above it so you don't get confused
    # One side to handle title and comments data
    if singleCol==True:
        if type(keys) is list and len(keys) == 1:
            key = keys[0]
        elif type(keys) is str:
            key = keys
        elif type(keys) is list and len(keys) >= 1:
            raise Exception("You specified `singleCol=True` and specified multiple keys. Please only specify one CSV key for single column example extraction.")
        elif keys is None:
            raise Exception("Please specify a key from which the function can generate examples based on a field in the dataset.")

        df = pd.read_csv(csvfile)
        if specificItem is not None:
            data = df.loc[df[key]==specificItem]['categorizations'].tolist()
            cats = []
            for i in range(len(data)):
                if type(data[i]) is str and fixDicts == True:
                    cats.append(read_dict(data[i])['labels'][0])
                elif type(data[i]) is dict and fixDicts == False:
                    cats.append(data[i]['labels'][0])
                elif type(data[i]) is str and fixDicts != True:
                    raise Exception("Column contains strings and `fixDicts` was not enabled. Please enable this or check the behavior of this function accordingly.")
                elif type(data[i]) is dict and fixDicts == True:
                    raise Exception("Column contains dictionaries and `fixDicts` was enabled. Please disable this or check the behavior of this function accordingly.")
                elif type(data[i]) is float:
                    cats.append("N/A")
            counts = Counter(cats)
            return {specificItem: counts}
        else:
            # Do for all unique columns
            allData = {}
            unique_fields = np.unique(df[key])
            for field in unique_fields:
                data = df.loc[df[key]==field]['categorizations'].tolist()
                cats = []
                for i in range(len(data)):
                    if type(data[i]) is str and fixDicts == True:
                        cats.append(read_dict(data[i])['labels'][0])
                    elif type(data[i]) is dict and fixDicts == False:
                        cats.append(data[i]['labels'][0])
                    elif type(data[i]) is str and fixDicts != True:
                        raise Exception("Column contains strings and `fixDicts` was not enabled. Please enable this or check the behavior of this function accordingly.")
                    elif type(data[i]) is dict and fixDicts == True:
                        raise Exception("Column contains dictionaries and `fixDicts` was enabled. Please disable this or check the behavior of this function accordingly.")
                    elif type(data[i]) is float:
                        cats.append("N/A")
                counts = Counter(cats)
                allData[field] = counts
            return allData
    elif not singleCol:
    # Another side to handle user data and other assorted things where multiple keys are necessary to parse categorizations (i.e., one is just a giant list that needs to be filtered, while the other is by individual field in the dataframe)
        df = pd.read_csv(csvfile)
        if specificItem:
            raise Exception("You passed a `specificItem` parameter without enabling `singleCol`. Please remove the parameter or ensure this is the correct behavior for your application.")
        if type(keys) is not list:
            keys = [keys]
        elif keys is None:
            raise Exception("Please specify which keys you would like to parse in the CSV.")
        allData = {}
        for key in keys:
            data = []
            if fixDicts:
                for i in range(len(df)):
                    if type(df[key][i]) is not float:
                        data.append(read_dict(df[key][i])['labels'][0])
                    else:
                        data.append("N/A")
            elif not fixDicts:
                for i in range(len(df)):
                    if type(df[key][i]) is not float:
                        data.append(df[key][i]['labels'][0])
                    else:
                        data.append("N/A")
            counts = Counter(data)
            cur_key = (key.split("_"))[0]
            allData[cur_key] = counts
        return allData

def getSingleDist(category_dict:dict,key=None):
    if len(category_dict.keys()) == 1:
        return list(category_dict.items())[0]
    elif key is not None:
        return (key,category_dict[key])
    else:
        raise Exception("The dictionary passed was not of length 1, and the `key` argument was not passed..")


if __name__ == '__main__':
    getCategoryDist('RITEC_studio_comments.csv',keys="studio_title",fixDicts=True,singleCol=True)