from elasticsearch import helpers, Elasticsearch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# this program may take about 20 minutes.
# how to run this program:
# 1. install ES (Elasticsearch) and run it
# 2. pip install nltk
# 3. download relative packages
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')

import time
start_timer = time.time()

#
# preapration for data, data path
#

path = "C:/Users/13418/Desktop/practice_place/ai_place/IR_place/wiki_movie_plots_deduped.csv"
df = pd.read_csv(path)
print( 'all movie data shape: ', df.shape )

# dataframe cols/features
def getCols(df):
    cols = []
    for col in df:
        cols.append(col)
    return cols

# print out the features of movie
cols = getCols(df)
print( "columns/features: ", cols )

#sample of 1000 articles, randomly
#num = 1000
#sample = df.sample(num, random_state=6)
sample = df
print('in this project, all movies will be uploaded for searching, it will takes about 20 minutes to run all this program once')

def basicInfo(df, verbose=True):
    RY_range = list( set( df['Release Year'] ) )
    origin_range = list( set( df['Origin/Ethnicity'] ) )
    genre_range = list( set( df['Genre'] ) )
    gen = set()
    for i in genre_range:
        if '/' in i:
            tmp = i.split('/')
            for w in tmp:
                gen.add(w.strip())
        elif ',' in i:
            tmp = i.split(',')
            for w in tmp:
                gen.add(w.strip())
        else:
            gen.add(i)
    genre_range = list( gen )
    if verbose:
        print("the basic info about those movie collection:")
        print("Release Year: \t", RY_range)
        print("Origin/Ethnicity: \t", origin_range)
        print("Genre: \t", genre_range)
    return RY_range, origin_range, genre_range

# print out the basic info about the data
# if basicInfo(sample, True),
# it will also print out the details about all features
basicInfo(sample, False)
print()


from datetime import datetime

# 
# indexing
# 

def createES(idx, dt):
    # connect to Elasticsearch
    print('create ES ...')
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    # create the index with doc type, put the mapping
    es.indices.delete(index=idx, ignore=[400, 404])
    es.indices.create(index=idx, ignore=400)
    return es

def add_mapping(es, idx, dt, mapping):
    es.indices.put_mapping(index=idx, doc_type=dt, body=mapping, include_type_name = True)
    return 

# return the uploaded documents' indexes
def uploadDocToES(es, sample, idx, dt):
    print( "indexing documents ..." )
    print( "upload a sample of ", len(sample), " articles with full text to Elasticsearch" )
    index_list = []
    for ind, row in sample.iterrows():
        my_doc = row.to_dict()
        es.index(index=idx, doc_type=dt, id=ind, body=my_doc) #, ignore=400)
        index_list.append( ind )
    return index_list

# for testing, to get the plot by index
def getPlot(es, idx, doc_idx):
    res = es.get(index=idx, id=doc_idx)
    test_plot = res['_source']['Plot']
    #print(test_plot) 
    return test_plot



# 
# Sentence Splitting, Tokenization and Normalization
#

# use ES to help to process entered query
# remove stopwords and Tokenization 
def tokenization(es, inputText, analyzer="english"):
    #analyzer = ['english'] # stop
    res = es.indices.analyze(body={"analyzer" : analyzer,"text" : inputText})
    tokens = []
    for i in res['tokens']:
        tokens.append( i['token'] )
    return tokens

# Sentence Splitting
def senSplit(es, inputText, analyzer="english"):
    print( "Sentence Splitting, Tokenization and Normalization ..." )
    sen_dic = {}
    s_counter = 1
    sentence_delimiter = '. '
    sentences = inputText.split(sentence_delimiter)
    for sentence in sentences:
        sentence = tokenization(es, sentence, analyzer)
        if len(sentence) > 0:
            sen_dic[s_counter] = sentence
            s_counter += 1
    return sen_dic


# update
# take the query as a whole sentence, no need to split the query as smaller sentences
def sen_split(es, inputText, analyzer="english"):
    print( "Sentence Splitting, Tokenization ..." )
    sentence = tokenization(es, inputText, analyzer)
    sen_dic = {}
    sen_dic[1] = sentence
    return sen_dic


#
# Selecting Keywords
#

# form word-set from your input text
def termSet(sen_dic):
    ws = set()
    for i in sen_dic:
        ws = ws.union( set(sen_dic[i]) )
    return ws

# calculate the term frequency for every sentence
def termFre(ws, sen):
    tf = dict.fromkeys(ws, 0)
    for i in sen:
        tf[i] = tf[i] + 1
    doc_len = len(sen)
    for i in tf:
        tf[i] = tf[i] / doc_len
    return tf

import math

# IDF, calculate the idf for every word/token
def termIDF(ws, sen_dic):
    N = len( sen_dic )
    idf = dict.fromkeys(ws, 0)
    for i in idf:
        c = 0
        for j in sen_dic:
            if i in sen_dic[j]:
                c = c + 1
                #rint(i, sen_dic[j])
        idf[i] = math.log( N/c )
        #print( i )
    return idf

# calculate the weight for every word in every document/sentence
# sen_dic, dict that includes many sentences split by inputText
def calWeight(sen_dic):
    ws = termSet(sen_dic)
    idf = termIDF(ws, sen_dic)
    weights = {}
    for i in sen_dic:
        sen = sen_dic[i]
        tf = termFre(ws, sen)
        wgt = {}
        for j in tf:
            w = tf[j] * idf[j]
            if w > 0: # only reserve the terms whose weight > 0
                wgt[j] = w
        # order by weight:
        #wgt = sorted(wgt.items(),key=lambda x:x[1],reverse=True)
        wgt = dict(sorted(wgt.items(), key=lambda item: item[1],reverse=True))
        #print(wgt)
        weights[i] = wgt        
    return weights

from collections import Counter
# select keywords
def selectKeys(sen_dic, top=10):
    print("Selecting Keywords ...")
    weights = calWeight(sen_dic)
    keys = set()
    for i in weights:
        c = Counter( weights[i] )
        L = len(weights[i])
        if L > top:
            L = top
        most_common = c.most_common(L)
        tmp = [key for key, val in most_common]
        keys = keys.union( set(tmp) )
    keys = list( keys )
    return keys


# here, POS Tagging are used to remove pronouns & some useless words such as what, when
# from the query if the query is long
# but to be honest, the improvement may not be very much

# !pip install nltk
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('tagsets')
def remove_pronouns(sentence):
    tags = {'CC': 'conjunction, coordinating', 'CD': 'numeral, cardinal', 'DT': 'determiner', 'EX': 'existential', 'FW': 'foreign word', 'IN': 'preposition', 'JJ': 'adjective', 'LS': 'list marker', 'MD': 'modal auxiliary', 'NN': 'noun', 'PDT': 'pre-determiner', 'POS': 'genitive marker', 'PR': 'pronoun', 'RB': 'adverb', 'RP': 'particle', 'SYM': 'symbol', 'UH': 'interjection', 'VB': 'verb', 'WDT': 'WH-determiner', 'WP': 'WH-pronoun', 'WRB': 'Wh-adverb'}
    words = sentence
    pos_tagged = nltk.pos_tag(words)
    new_list = []
    for w, tag in pos_tagged:
        #print(w, tag)
        if tag[:2]=='PR' or tag[:1]=='W' or tag[:1]=='M':
            #print(w)
            pass
        else:
            new_list.append(w)
    return new_list

# update
def select_keys(sen_dic, top=10, pos_tag=True):
    print("Selecting top ", top, " Keywords ...")
    sentence = sen_dic[1]
    if pos_tag:
        sentence = remove_pronouns(sentence)   
    wds = sentence
    if len(wds) <= top:
        return wds
    #
    ws = termSet(sen_dic)
    tf = termFre(ws, wds)
    c = Counter( tf )
    L = len( tf )
    if L > top:
        L = top
    most_common = c.most_common(L)
    tmp = [key for key, val in most_common]
    ks = list(tmp)
    return ks

#
# Stemming or Morphological Analysis
#

def add_setting(es, idx, setting):
    print( "Stemming or Morphological Analysis ... " )
    #print( "here is the used analysis filter & analyzer for following search: \t", setting )
    # close first, then add settings, then open
    es.indices.close(index=idx)
    es.indices.put_settings(index=idx, body=setting )
    # es.indices.put_mapping(index=my_index, doc_type=my_doc_type, body=mapping, include_type_name = True)
    es.indices.open(index=idx)
    return 


#
# Searching
#

# generate query to search, gievn input Text
def generateQuery(queryText, origin="", genre="", yearFrom=1900, yearTo=2022, searchContent=[ 'Title', 'Plot', 'Director', 'Cast', 'Wiki Page']):
    if len(queryText) == 0: # return all
        query_body = { "query":{ "match_all":{} } }
        return query_body
    
    # basic query
    query_body = { "query": {  "bool": {  "must": [ { "multi_match": { "query": queryText, "fields" : searchContent } } ],
          "filter": [ { "range": { "Release Year":{"gt":yearFrom, "lt":yearTo} }}  ] } } }
    
    # when user decide certain fields such as Origin/Ethnicity, Genre
    if len(origin) > 0:
        query_body["query"]["bool"]["filter"].append( { "match": { 'Origin/Ethnicity':  origin }} )
    if len(genre) > 0:
        query_body["query"]["bool"]["filter"].append( { "match": { 'Genre':  genre }} )
    return query_body


# search, print the top 10 recall results
def searching(es, idx, querybody, top=10, verbose=False):
    print( "Searching ..." )
    result = es.search(index=idx, body=querybody)
    recallNum = result['took']
    recallContent = result['hits']['hits']
    if top>recallNum:
        top = recallNum
    print( "find the top ", top, " most relevant results: \n" )
    idx_list = []
    for it in recallContent[:top]:
        print( "index: ", it['_id'], "\t relevant score: ", it['_score'] )
        idx_list.append(it['_id'])
        if verbose:
            content = it['_source']
            print( "Release Year: ", content['Release Year'], "\t Origin/Ethnicity: ", content['Origin/Ethnicity'] )
            print( "Genre: ",  content['Genre'] )
            print( "Title: ", content['Title'] )
            print( "Plot: ", content['Plot'][:100], "..." )
            print(  ) 
    return idx_list

# for testing, to get movie detail by index
def getDetail(sample, idx):
    movie_detail = sample.loc[ idx ]
    print( "Release Year: ", movie_detail['Release Year'], "\t Origin/Ethnicity: ", movie_detail['Origin/Ethnicity'] )
    print( "Genre: ",  movie_detail['Genre'] )
    print( "Title: ", movie_detail['Title'] )
    print( "Director: ",  movie_detail['Director'] )
    print( "Cast: ", movie_detail['Cast'] )
    print( "Wiki Page: ",  movie_detail['Wiki Page'] )
    print( "Plot: ", movie_detail['Plot'] )
    return movie_detail


# test 
#print(es1.indices.get_mapping(index=my_index))
#print(es1.indices.get_settings(index=my_index))


# there may be some NaN values in the sample data
sample = sample.replace(np.nan, '', regex=True) # deal with NaN

# stemming / morph 
def get_setting():
    setting2 = {
      "settings": {
        "analysis": {
            "filter": {
                "english_stop": {
                  "type":       "stop",
                  "stopwords":  "_english_"
                },
                "light_english_stemmer": {
                  "type":       "stemmer",
                  "language":   "light_english" 
                },
                "english_possessive_stemmer": {
                  "type":       "stemmer",
                  "language":   "possessive_english"
                }
              },
              "analyzer": {
                "my_analyzer01": { 
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [ "lowercase", "asciifolding"]
                },
                "my_analyzer02": { 
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [ "lowercase", "asciifolding", "english_stop"]
                },
                "my_analyzer03": { 
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [ "lowercase", "asciifolding", "english_stop", "light_english_stemmer"]
                },
                "my_analyzer04": { 
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": [ "lowercase", "asciifolding", "english_stop", "english_possessive_stemmer"]
                }
              }
        }
      }
    }
    return setting2



# the document is mapping as the following structure
def get_mapping(num=2):
    analyzer_list = ["whitespace", "my_analyzer01", "my_analyzer02", "my_analyzer03", "my_analyzer04"]
    use_analyzer = analyzer_list[num]
    mapping = {
        "properties": {
            "id":{
                "type":"long"
            },
            "Release Year": {
                "type": "long"
            },
            'Title':{
                "type": "text",
                 
                "analyzer": use_analyzer
            },
            'Origin/Ethnicity':{
                "type": "text",
                 
                "analyzer": use_analyzer
            },
            'Director': {
                "type": "text",
                 
                "analyzer": use_analyzer
            },
            'Cast': {
                "type": "text",
                 
                "analyzer": use_analyzer
            },
            'Genre': {
                "type": "text",
                 
                "analyzer": use_analyzer
            },
            'Wiki Page': {
                "type": "keyword"
            },
            'Plot': {
                "type": "text",
                 
                "analyzer": use_analyzer
            }
        }
    }    
    return mapping



###  Building a Test Collection
        # three information needs
        # a sample queries for each
def test_collection():
    # information need 1, specific in Origin/Ethnicity, years, 
        # binary model
        # retrieve the American movies, from 1990 to 2010, about drama
    query_1 = {}
    query_1['yearFrom'] = 1990
    query_1['yearTo'] = 2010
    query_1['origin'] = 'American'
    query_1['genre'] = 'drama'
    query_1['inputText'] = 'american drama'
    
    # information need 2, specific in the plot/topic
        # vector model
        # retrieve james bond movies or some adventure mission films, about espionage, spying
    query_2 = {}
    query_2['yearFrom'] = 1900 # default value, just very early
    query_2['yearTo'] = 2021  # default value, just very recent
    query_2['origin'] = '' # no restraint
    query_2['genre'] = '' # no restraint
    query_2['inputText'] = 'james bond movies or some adventure mission films, about espionage, spying '
    
    # information need 3, specific in Origin/Ethnicity, years, & the plot/topic
        # binary + vector
        # retrieve james bond movies or some adventure mission films, about espionage, spying, 
            # from 1990 to 2018, 
            # origin: American or British, 
            # genre: drama, action, spy, romance
    query_3 = {}
    query_3['yearFrom'] = 1990 
    query_3['yearTo'] = 2018
    query_3['origin'] = 'American, British' 
    query_3['genre'] = 'drama, action, spy, romance' 
    query_3['inputText'] = 'james bond movies or some adventure mission films, about espionage, spying '
    
    return query_1, query_2, query_3



# split the entered query
    # split_mode='stop', split_mode='english'
# extract from query at most words:
    # key_num=10
# use pos_tag to remove the pronouns from the query such as: he, him, his 
    # pos_tag=True, pos_tag=False
# origin, genre, yearFrom, yearTo
    # limit the searching range of the movie    
# returned results after searching
    # retrieved=10
# print out the detail
    # verbose=False
def search_engine(sample, query_test, num=0, split_mode='stop', key_num=10, pos_tag=True, retrieved=10, verbose=False):
    # index name, doc type
    my_index = 'ir_hw'
    my_doc_type = 'movie'
    print( "index name: ", my_index, "\t doc type: ", my_doc_type)
    # create 
    es1 = createES(my_index, my_doc_type)
    # add setting: analyzer
    setting2 = get_setting()
    add_setting(es1, my_index, setting2)
    # add mapping & upload movies
    mapping = get_mapping(num)
    add_mapping(es1, my_index, my_doc_type, mapping)
    index_list = uploadDocToES(es1, sample, my_index, my_doc_type)
    # process entered query
    inputText = query_test['inputText']
    sen_dic = sen_split(es1, inputText, split_mode)
    # extract from query
    keys = select_keys(sen_dic, key_num, pos_tag)
    keys = " ".join(keys)
    # form query & search
    queryText = keys
    searchContent = [ 'Title', 'Plot', 'Director', 'Cast', 'Wiki Page']
    querybody = generateQuery(queryText, query_test['origin'], query_test['genre'], query_test['yearFrom'], query_test['yearTo'])
    idx_list = searching(es1, my_index, querybody, retrieved, verbose)
    print()    
    return idx_list


    # num = 0, num = 1, only whitespace
    # num = 2, only use stopword
    # num = 3, num = 4, use stopword + stemmer
def IR_systems(query_test, stms, sample):
    col = {}
    for i in stms:
        print("-----------------")
        num = i
        idx_list = search_engine(sample, query_test, num, split_mode='stop', key_num=20, pos_tag=True, retrieved=10, verbose=False)
        #print(idx_list)
        col[i]= idx_list    
    return col


#inputText = input("please input your query: \n")
#print()


query_1, query_2, query_3 = test_collection()
print('test collection with 3 queries: \t')
print('query 1: \t', query_1)
print('query 2: \t', query_2)
print('query 3: \t', query_3)
print()


# two IR systems with different configurations/parameters
# 0, means mapping by 'whitespace': 
    #  "tokenizer": "standard",
    #  "filter": [ "whitespace"]
# 1, means mapping by 'my_analyzer01': 
    #  "tokenizer": "standard",
    #  "filter": [ "lowercase", "asciifolding"]
# 2, means mapping by 'my_analyzer02': 
    #  "tokenizer": "standard",
    #  "filter": [ "lowercase", "asciifolding", "english_stop"]
# 3, means mapping by 'my_analyzer03': 
    #  "tokenizer": "standard",
    #  "filter": [ "lowercase", "asciifolding", "english_stop", "light_english_stemmer"]
# 4, means mapping by 'my_analyzer04': 
    #  "tokenizer": "standard",
    #  "filter": [ "lowercase", "asciifolding", "english_stop", "english_possessive_stemmer"]
stms = [2, 3] 
# but you could also try [0, 3], [1, 3], [0, 4], [1, 4], [2, 4]


query_test = query_1
print('searching query 1 by two IR systems: \t-----------------')
col1 = IR_systems(query_test, stms, sample)

query_test = query_2
print('searching query 2 by two IR systems: \t-----------------')
col2 = IR_systems(query_test, stms, sample)

query_test = query_3
print('searching query 3 by two IR systems: \t-----------------')
col3 = IR_systems(query_test, stms, sample)


# col, includes # top 10 retrieved top-10 results from 2 IR systems
def pooling(col):
    pool = []
    for i in col:
        tmp = col[i]
        for it in tmp:
            if it not in pool:
                pool.append(it)
    return pool

def assess_relevance(query_num):
    # binary relevance judgements
    # either relevant or non-relevant
    
    # after eye-checking the pooling results, 
    # namely, judge every document from the pooling
    # the following indexed are relevant to the given query
    
    # for query 1: retrieve the American movies, from 1990 to 2010, about family drama
        # binary model
    if query_num==1:
        return ['11970', '12770', '12511', '13648', '14289', '14649', '15428', '14109', '15156', '11866']
    
    # for query 2: retrieve james bond movies or some adventure mission films, about espionage, spying
        # vector model
    if query_num==2:
        # 8169, 3679, 29364, 18664, 19672, 5340, 18684, (17151, 10194), 13057, (4794, 15414)
        return ['8169', '3679', '29364', '18664', '19672', '5340', '18684', '13057']
    
    # for query 3: retrieve james bond movies or some adventure mission films, about espionage, spying, 
        # from 1990 to 2018, 
        # origin: American or British, 
        # genre: drama, action, spy, romance
    if query_num==3:
        # 14852, 20612, 15507, 21001, 20801, 20738, 20663, 14910, 15939, 12338
        return ['14852', '20612', '15507', '21001', '20801', '20738', '20663', '14910', '15939', '12338']
    return []

def evaluation(query_num, col):
    # eye-checking results as the criteria
    criteria = assess_relevance(query_num)
    
    # P@5
    p5 = {}
    for i in col:
        ir = col[i][:5] # get the top 5
        tmp = []
        s = 0
        count = 0
        for it in ir:
            count = count + 1
            if it in criteria:
                s = s + 1
                tmp.append(s/count)
            else:
                tmp.append(s/count)
        p5[i] = tmp            
    
    # R@5
    r5 = {}
    L = len(criteria)
    for i in col:
        ir = col[i][:5] # get the top 5
        tmp = []
        s = 0
        for it in ir:
            if it in criteria:
                s = s + 1
                tmp.append(s/L)
            else:
                tmp.append(s/L)
        r5[i] = tmp       
    
    return p5, r5


# each one corresponding to a query,
# pooling the retrieved results from 2 IR systems that have different parameters
# then eye-check every document from the pooling
p1 = pooling(col1)
p2 = pooling(col2)
p3 = pooling(col3)
print('Pooling 1: ')
print(p1)
print('Pooling 2: ')
print(p2)
print('Pooling 3: ')
print(p3)
print()
# getDetail, to check every document detail by index
# getDetail(sample, 17281)
"""
p1, len(p1) # all are relevant
p2, len(p2) # relevant: 8169, 3679, (29364), 18664, 19672, 5340, 18684, (17151, 10194), 13057, (4794, 15414)
p3, len(p3) # relevant: 14852, 20612, 15507, 21001, 20801, 20738, 20663, 14910, 15939, 12338
"""

query_num = 1
p5, r5 = evaluation(query_num, col1)
print('2 IR systems searching with query 1: ')
print('P@5\t', p5)
print('R@5\t', r5)

query_num = 2
p5, r5 = evaluation(query_num, col2)
print('2 IR systems searching with query 2: ')
print('P@5\t', p5)
print('R@5\t', r5)

query_num = 3
p5, r5 = evaluation(query_num, col3)
print('2 IR systems searching with query 3: ')
print('P@5\t', p5)
print('R@5\t', r5)

print()
print()
end_timer = time.time()
time_interval = end_timer - start_timer
print("============================================")
print('time used: ', time_interval)
print("Thank you!")

