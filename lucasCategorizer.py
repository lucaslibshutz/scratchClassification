from transformers import pipeline, Pipeline, PreTrainedModel, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import pandas as pd
import string
import gc
import re
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from huggingface_hub import HfFolder

HfFolder.save_token('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')


#! Check if NLTK stuff is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def find_divisors(n):
    if type(n) is list:
        n = len(n)
    divisors = []
    for i in range(1,n+1):
        if n%i == 0:
            divisors.append(i)
    return divisors

def clear_vram(m,tokenizer=None):
    if isinstance(m,Pipeline):
        m.model.to('cpu')
        del m.model
        del m.tokenizer
        del m
    elif isinstance(m,PreTrainedModel) or isinstance(m,SentenceTransformer):
        m.to('cpu')
        del m
        if tokenizer:
            del tokenizer
    else:
        print("Unknown object type")
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def contains_chinese(s):
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(s))

def findLanguages(texts: dict):
    texts = [[ind,x] for ind,x in texts.items() if type(x) is not float and x is not None and (np.any([y in x for y in string.ascii_lowercase]) or contains_chinese(x))]
    detector = pipeline("text-classification",model="papluca/xlm-roberta-base-language-detection",device=0)
    indices = np.array(texts)[:,0].tolist()
    entries = np.array(texts)[:,1].tolist()
    foundLangs = pd.DataFrame(detector(entries,batch_size=50))['label'].tolist()
    clear_vram(detector)
    withTargets = []
    for i in range(len(texts)):
        withTargets.append([int(indices[i]),entries[i],foundLangs[i]])
    return withTargets

def translateData(texts,max_batch_size=None):
    withTargets = findLanguages(texts)

    already_en = [[x[0],x[1]] for x in withTargets if x[2]=='en']
    not_en = [[x[0],f"<2en> {x[1]}"] for x in withTargets if x[2]!='en']
    nEn_indices = [x[0] for x in not_en]
    to_translate = [x[1] for x in not_en]

    divList = find_divisors(not_en)
    if not max_batch_size:
        batch_size = divList[np.argmax([x for x in divList if x < 100])]
    else:
        batch_size = max_batch_size

    model_name = 'jbochi/madlad400-3b-mt'
    model = T5ForConditionalGeneration.from_pretrained(model_name,device_map='auto').to('cuda')
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    translations = []
    for i in tqdm(range(0,len(to_translate),batch_size)):
        batch = to_translate[i:i+batch_size]

        inputs = tokenizer(batch,return_tensors='pt',padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs)

        # Decode translations
        decoded = tokenizer.batch_decode(outputs,skip_special_tokens=True)
        translations.extend(decoded)

        inputs.to('cpu')
        outputs.to('cpu')
        del inputs
        del outputs
    translated = [[x,y] for x,y in zip(nEn_indices,translations)]
    finalList = sorted(already_en+translated,key=lambda x:x[0])
    t_indices = [x[0] for x in finalList]
    t_entries = [x[1] for x in finalList]
    clear_vram(model,tokenizer)
    return (t_indices,t_entries)

def clusterText(texts,n_clusters):
    words = []
    for text in texts:
        for word in text.split(' '):
            words.append(word)
    
    def process(texts):
        tokens = [token.lower() for token in texts if token not in stop_words and token not in string.punctuation and "http" not in token]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens
    
    def extract_terms(clustered_words,clustered_embeddings):
        cluster_labels = []
        for id in clustered_words:
            words = clustered_words[id]
            embeddings = clustered_embeddings[id]

            word_counts = Counter(words)

            centroid = np.mean(embeddings,axis=0)

            similarities = util.pytorch_cos_sim(centroid,embeddings).numpy().flatten()

            combined_scores = {word: word_counts[word] * similarities[idx] for idx,word in enumerate(words)}
            sorted_words = sorted(combined_scores,key=combined_scores.get,reverse=True)
            top_terms = sorted_words[:10]
            cluster_labels.append(top_terms)
        return cluster_labels

    pWords = process(words)

    embedding_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5")
    embeddings = embedding_model.encode(pWords,show_progress_bar=True)
    clear_vram(embedding_model)

    kmeans = KMeans(n_clusters=n_clusters,random_state=42)
    kmeans.fit(embeddings)
    # cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    clustered_words = {i: [] for i in range(n_clusters)}
    clustered_embeddings = {i: [] for i in range(n_clusters)}
    for word,embedding,label in zip(pWords,embeddings,labels):
        clustered_words[label].append(word)
        clustered_embeddings[label].append(embedding)
    
    cluster_labels = extract_terms(clustered_words,clustered_embeddings)

    return cluster_labels

def genCategories(clustered_labels):
    labels_to_use = [x[0] for x in clustered_labels]
    model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    pipe = pipeline("text-generation",model=model_id,model_kwargs={"torch_dtype":torch.bfloat16},device_map='auto')
    messages = [
        {"role": "system", "content": "You are a bot that takes in a list of words, and for each one gives a one-word category description. For example, the word 'soon' would become 'Time/Timing'. If one of the categories is 'unknown' or 'ambiguous', DO NOT include it in the final list of categories. Once done, make sure that all of the categories are unique from one another. Return **only** the categories as a bulleted list with only the categories."},
        {'role': 'user', 'content': 'Categorize the following'+str(labels_to_use)}
    ]
    outputs = pipe(messages,max_new_tokens=256)
    finalStuff = outputs[0]['generated_text'][-1]['content'].split('\n')
    fixed = [x[2:] for x in finalStuff if 'none' not in x.lower()]
    fixed = np.unique([x for x in fixed if 'ambiguous' not in x.lower() and 'unknown' not in x.lower()]).tolist()
    clear_vram(pipe)
    return fixed
    
def classifyData(data,categories,batch_size=None):
    classifier = pipeline("zero-shot-classification",model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",device=0)
    longData = [x for x in data if len(x) >= 3]
    if batch_size:
        classified = classifier(longData,categories,multi_label=True,batch_size=batch_size)
    else:
        classified = classifier(longData,categories,multi_label=True)
    clear_vram(classifier)
    return classified


def combineAndExport(dataframe,outputs,outname,multiField=False,pickle=False):
    if multiField:
        myColumns = {}
        for key in outputs.keys():
            zList = np.zeros(len(dataframe),dtype=object)
            for output in outputs[key]:
                zList[output[0]] = output[1]
            for i in range(len(zList)):
                if zList[i] == 0:
                    zList[i] = np.nan
            myColumns[key] = zList
        dfA = dataframe.assign(**myColumns)
    else:
        myColumn = {'categorizations': []}
        zList = np.zeros(len(dataframe),dtype=object)
        for output in outputs:
            zList[output[0]] = output[1]
        for i in range(len(zList)):
            if zList[i] == 0:
                zList[i] = np.nan
        myColumn['categorizations'] = zList
        dfA = dataframe.assign(**myColumn)
    if pickle:
        dfA.to_pickle(outname,index=None,compression=None)
    else:
        dfA.to_csv(outname,index=None,compression=None)
    print(f"Dataframe exported as {outname}")

def exportData(dataframe,allCategorizations,output_file,pickle=False):
    myColumns = {}
    for key in allCategorizations.keys():
        zList = np.zeros(len(dataframe),dtype=object)
        for output in allCategorizations[key]:
            zList[output[0]] = output[1:]
        for i in range(len(zList)):
            if zList[i] == 0:
                zList[i] = ["N/A",1.0]
        myColumns[key] = zList
    dfA = dataframe.assign(**myColumns)
    if pickle:
        dfA.to_pickle(output_file)
    else:
        dfA.to_csv(output_file,index=None,compression=None)


def scratchClassify(csvfile,column,base,identifiers,output_file):
    df = pd.read_csv(csvfile)
    if type(column) is not str:
        raise Exception("The column to classify was not passed in as a string.")
    if base is None and identifiers is None:
        raise Exception("Please choose what kind of classification you would like: either just the base column or based off of an identifier.")
    elif base is None:
        base = False
    elif identifiers is not None and type(identifiers) is str:
        identifiers = [identifiers]
    
    allCategorizations = {}
    # Whether or not to perform the base categorization
    if base == True:
        data = df[column].to_dict()
        indices,translated = translateData(data,max_batch_size=10)
        clustered_labels = clusterText(translated,n_clusters=8)
        categories = genCategories(clustered_labels)
        outputs = [[x['labels'][0],x['scores'][0]] for x in classifyData(translated,categories,batch_size=50)]
        finalOutputs = [[index] + output for index,output in zip(indices,outputs)]
        allCategorizations[f'base_{column}_categorization'] = finalOutputs
    if identifiers:
        for identifier in identifiers:
            fields = np.unique(df[identifier])
            allData = []
            for field in fields:
                data = df.loc[df[identifier]==field][column].to_dict()
                indices,translated = translateData(data,max_batch_size=25) #! change these once done with A100
                clustered_labels = clusterText(translated,n_clusters=8)
                categories = genCategories(clustered_labels)
                outputs = [[x['labels'][0],x['scores'][0]] for x in classifyData(translated,categories,batch_size=50)]
                finalOutputs = [[index]+ output for index,output in zip(indices,outputs)]
                allData.extend(finalOutputs)
            allData = sorted(allData,key=lambda x: x[0])
            allCategorizations[f'{identifier}_{column}_categorization'] = allData
        
    # Export data
    exportData(df,allCategorizations,output_file=output_file)