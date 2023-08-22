import sys

import pandas as pd
import numpy as np
import nltk
import re
import stringggggg
import csv
from nameparser.parser import HumanName
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import numpy.linalg as LA
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import PorterStemmer
from autocorrect import Speller
import datefinder
from num2words import num2words
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import scipy.sparse
from pprint import pprint
from pprint import pprint
from gensim import interfaces, utils, matutils
import gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import spacy



person_names = []

cisi_text = ""
f1 = open('d:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\CISI\cisi.txt', 'r')
for line in f1:
    cisi_text = cisi_text + line
cisi_corpus = cisi_text.split('\n')

cacm_text= ""
f2 = open('D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\cacm\cacm.txt', 'r')
for line in f2:
    cacm_text = cacm_text + line
cacm_corpus = cacm_text.split('\n')


cisi_dictionary = {
    '.I': [],
    '.C' : [],
    '.X' : []
}
cacm_dictionary = {
    '.I': [],
    '.C' : [],
    '.X' : []
}


def cisi_to_chunk(chunk):
    global person_names
    for sentence in chunk.split('\n'):
        if ".I" in sentence:
            start_index=chunk.find(".I")
            lword=len(".I")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+2
            required_string= chunk[required_string_index:chunk.find(".T")]
            cisi_dictionary[".I"].append(required_string)
        if ".T" in sentence:
            start_index=chunk.find(".T")
            lword=len(".T")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+2
            required_string= chunk[required_string_index:chunk.find(".X")]
            cisi_dictionary[".C"].append(required_string)
        if ".A" in sentence:
            start_index=chunk.find(".A")
            lword=len(".A")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+3
            required_string= chunk[required_string_index:chunk.find(".W")]
            person_names.append(required_string)
        if ".X" in sentence:
            start_index=chunk.find(".X")
            lword=len(".X")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+3
            required_string= chunk[required_string_index:chunk.find("\n.I")]
            cisi_dictionary[".X"].append(required_string)


isThereAnI = True
chunks_list = [None] * 1500
i = 0
for sentence in cisi_corpus:
    if ".I " in sentence:
        if isThereAnI:
            chunks_list[i]= ' '
            isThereAnI = False
        else:
            cisi_to_chunk(chunks_list[i])
            i = i + 1
            chunks_list[i] = ' '
    chunks_list[i] = chunks_list[i] + sentence + ' '


cisi_df = pd.DataFrame(data=cisi_dictionary)
# pd.set_option('max_columns', 5)
pd.set_option('max_colwidth', 10000)
cisi_df.columns = ['id','content','references']


def cacm_to_chunk(chunk):
    global person_names
    for sentence in chunk.split('\n'):
        if ".I" in sentence:
            start_index=chunk.find(".I")
            lword=len(".I")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+2
            required_string= chunk[required_string_index:chunk.find(".T")]
            cacm_dictionary[".I"].append(required_string)
        if ".T" in sentence:
            start_index=chunk.find(".T")
            lword=len(".T")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+2
            required_string= chunk[required_string_index:chunk.find(".N")]
            cacm_dictionary[".C"].append(required_string)
        if ".A" in sentence:
            start_index=chunk.find(".A")
            lword=len(".A")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+3
            required_string= chunk[required_string_index:chunk.find(".N")]
#             cacm_dictionary[".C"].append(required_string)
            person_names.append(required_string)
        if ".X" in sentence:
            start_index=chunk.find(".X")
            lword=len(".X")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+3
            required_string= chunk[required_string_index:chunk.find("\n.I")]
            cacm_dictionary[".X"].append(required_string)


isThereAnI = True
chunks_list = [None] * 3300
i = 0
for sentence in cacm_corpus:
    if ".I " in sentence:
        if isThereAnI:
            chunks_list[i]= ' '
            isThereAnI = False
        else:
            cacm_to_chunk(chunks_list[i])
            i = i + 1
            chunks_list[i] = ' '
    chunks_list[i] = chunks_list[i] + sentence + ' '



cacm_df = pd.DataFrame(data=cacm_dictionary)
pd.set_option('max_colwidth', 10000)
cacm_df.columns = ['id','content','references']

# first round cleaning
# defining the names

person_list = []
persons = person_list
def clean_text_round1(text):
    global person_names
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)
    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []
    for person in person_list:
        person_split = person.split(" ")
        for name in person_split:
            if wordnet.synsets(name):
                if(name in person):
                    persons.remove(person)
                    break
    person_names = person_names + persons
#     print(person_names)
    return text
round1 = lambda x: clean_text_round1(x)




#second cleaning round
#cleaning the additional \n and \t
#Apply a second round of text cleaning techniques
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
commom_words = open('D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\cacm\common_words', 'r').read().split()

def clean_text_round2(text):
    new_text = ' '
    text = text.lower()
    text = re.sub('[''""_]',' ',text)
    text = re.sub('\t',' ',text)
    text = re.sub('\n',' ',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ',text)
    word_tokens = word_tokenize(text)
    for w in word_tokens:
#         print(w)
        if w not in commom_words and w not in stop_words:
            new_text = new_text + w +' '
    return new_text
round2 = lambda x: clean_text_round2(x)


#third cleaning round
#working on auto correct
def contain_in_names_list(name):
    found = False
    for person_name in person_names:
        if person_name.find(name) != -1:
            found = True
    return found

def clean_text_round3(text):
    new_text = ''
    words = word_tokenize(text)
    spell = Speller(lang='en')
    for w in words:
        if(not contain_in_names_list(w)):
            w = spell(w)
            new_text += w + ' '
        else:
            new_text += w + ' '
    return new_text
round3 = lambda x: clean_text_round3(x)


#forth cleaning round
#working on dates finding
def clean_text_round4(text):
    text = sent_tokenize(text)
    required_text = ""
    for sentence in text:
        matches = datefinder.find_dates(sentence,source = True)
        if matches is not None:
            try:
                for match in matches:
                    required_year = match[0].year
                    required_date =str(required_year)
                    temp_date = match[1]
                    sentence = sentence.replace(temp_date,required_date,1)
            except:
                print("error with date")
        required_text = required_text + sentence
    return required_text
round4 = lambda x: clean_text_round4(x)


#fifth cleaning round
#Apply lemmitization round of text cleaning techniques
#which is from parallel to singular
def clean_text_round5(text):
    new_text = ''
    lemma = nltk.wordnet.WordNetLemmatizer()
    words = word_tokenize(text)
    for w in words:
        w=lemma.lemmatize(w,pos='v')
        w = lemma.lemmatize(w)
        new_text += w + ' '
    return new_text
round5 = lambda x: clean_text_round5(x)


def clean_text_round6(text):
    new_text = ''
    words = word_tokenize(text)
    for w in words:
        if w.isnumeric():
            w = num2words(int(w))
        if(re.search(r'\d',w)):
            w = ' '
        new_text += w + ' '
    return new_text
round6 = lambda x: clean_text_round6(x)



#seventh cleaning round
#Apply Stemming round of text cleaning techniques
#which is from past to current verbs

def clean_text_round7(text):
    ps = PorterStemmer()
    new_text = ''
    words = word_tokenize(text)
    for w in words:
        w = ps.stem(w)
        new_text += w + ' '
    return new_text
round7 = lambda x: clean_text_round7(x)


cisi_df.content = cisi_df.content.apply(round1)
cisi_df.content = cisi_df.content.apply(round2)
# cisi_df.content = cisi_df.content.apply(round3)
cisi_df.content = cisi_df.content.apply(round4)
cisi_df.content = cisi_df.content.apply(round5)
cisi_df.content = cisi_df.content.apply(round6)
cisi_df.content = cisi_df.content.apply(round7)



vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(cisi_df.content)
columns = vectorizer.get_feature_names_out()
data_dtm_2 = pd.DataFrame(vectors.todense(),columns = columns)
pd.set_option('max_colwidth', 100)
pd.set_option('max_columns', 10000)


cacm_df.content = cacm_df.content.apply(round1)
cacm_df.content = cacm_df.content.apply(round2)
#cacm_df.content = cacm_df.content.apply(round3)
cacm_df.content = cacm_df.content.apply(round4)
cacm_df.content = cacm_df.content.apply(round5)
cacm_df.content = cacm_df.content.apply(round6)
cacm_df.content = cacm_df.content.apply(round7)

vectorizer = TfidfVectorizer()
cacm_vectors = vectorizer.fit_transform(cacm_df.content)
data_dtm_cacm = pd.DataFrame(cacm_vectors.todense(),columns = vectorizer.get_feature_names_out())
pd.set_option('max_colwidth', 100)
pd.set_option('max_columns', 10000)

#Reading
#queries
#file


# reading all the queries
# E:\python\cisi_queries
cisi_queries = ""
cisi_queries_file = open('D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\CISI\cisi_queries.txt', 'r')
for line in cisi_queries_file:
    cisi_queries = cisi_queries + line
cisi_queries_corpus = cisi_queries.split('\n')

cacm_queries = ""
cacm_queries_file = open('D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\cacm\cacm_queries.txt', 'r')
for line in cacm_queries_file:
    cacm_queries = cacm_queries + line
cacm_queries_corpus = cacm_queries.split('\n')


cisi_queries = []
cacm_queries = []


def query_to_chunk(chunk,queries_list):
    for sentence in chunk.split('\n'):
        if ".W" in sentence:
            start_index=chunk.find(".W")
            lword=len(".W")
            extracted_string= chunk[start_index:start_index+lword]
            required_string_index = chunk.index(extracted_string)+2
            required_string= chunk[required_string_index:chunk.find(".I") - 2]
            required_string = re.sub('\n',' ',required_string)
            queries_list.append(required_string)


isThereAnW = True
chunks_list_2 = [' '] * 3000
i = 0
for sentence in cisi_queries_corpus:
#     print(sentence + "\n")
    if ".W" in sentence:
        if isThereAnW:
            chunks_list_2[i]= ' '
            isThereAnW = False
        else:
            query_to_chunk(chunks_list_2[i],cisi_queries)
            i = i + 1
            chunks_list_2[i] = ' '
    chunks_list_2[i] = chunks_list_2[i] + sentence


isThereAnW = True
chunks_list_3 = [' '] * 100
i = 0
for sentence in cacm_queries_corpus:
#     print(sentence + "\n")
    if ".W" in sentence:
        if isThereAnW:
            chunks_list_3[i]= ' '
            isThereAnW = False
        else:
            query_to_chunk(chunks_list_3[i],cacm_queries)
            i = i + 1
            chunks_list_2[i] = ' '
    chunks_list_3[i] = chunks_list_3[i] + sentence


cisi_answers_csv = pd.read_csv("D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\CISI\cisi_answers_csv.csv")
cisi_answers_csv.columns = ['query_id','document_id']
pd.options.display.max_rows = 3100


cisi_answers_dictionary = {}
count = 1
cisi_answers_dictionary[count] = []
for i in cisi_answers_csv.values:
    if ((i[0] == count)):
        cisi_answers_dictionary[count].append(i[1])
    else:
        count = count+1
        cisi_answers_dictionary[count] = []
        cisi_answers_dictionary[count].append(i[1])


import pandas as pd
cacm_answers_csv = pd.read_csv("D:\\fifth\IR\IR\IR_Project_datasets-20220522T051310Z-001\IR_Project_datasets\cacm\cacm_answers_csv.csv")
cacm_answers_csv.columns = ['query_id','document_id']
pd.options.display.max_rows = 3100

cacm_answers_dictionary = {}
count = 1
cacm_answers_dictionary[count] = []
for i in cacm_answers_csv.values:
    if ((i[0] == count)):
        cacm_answers_dictionary[count].append(i[1])
    else:
        count = count+1
        cacm_answers_dictionary[count] = []
        cacm_answers_dictionary[count].append(i[1])



cx = lambda a, b : np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def matching_function(corpus, queries, answers):
    output = {}
    vectorizer = TfidfVectorizer()
    train_set = corpus  # Documents
    test_set = queries
    new_test_set = []
    for test in test_set:
        test = clean_text_round1(test)
        test = clean_text_round2(test)
        test = clean_text_round3(test)
        test = clean_text_round5(test)
        test = clean_text_round6(test)
        test = clean_text_round7(test)
        new_test_set.append(test)

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(new_test_set).toarray()

    i = 1;
    for testV in testVectorizerArray:
        d_cosines = []
        for vector in trainVectorizerArray:
            cosine = cx(vector, testV)
            d_cosines.append(cx(vector, testV))
        number = len(answers.get(i))
        out = np.array(d_cosines).argsort()[-number:][::-1]
        output[i] = out
        if (i < len(answers)):
            i = i + 1

    for out in output:
        output[out].sort()

    for out in output:
        i = 0
        for num in output[out]:
            output[out][i] = num + 1
            i = i + 1
    #     print(output)
    return output


def calculate_measures(true_answers, test_answers, document_size):
    accurecy_measures = {'true_answers': [], 'test_answers': [], 'precision': [], 'recall': []}
    counter = 1
    for true_answer, test_answer in zip(true_answers, test_answers):
        #         print(counter)
        accurecy_measures['true_answers'].append(counter)
        accurecy_measures['test_answers'].append(counter)
        arr_1 = [0] * document_size
        k = 0
        for i in arr_1:
            for j in true_answer:
                if k == j:
                    arr_1.insert(j, 1)
            k = k + 1
        arr_2 = [0] * document_size
        m = 0
        for i in arr_2:
            for j in test_answer:
                if m == j:
                    arr_2.insert(j, 1)
            m = m + 1

        precision = precision_score(arr_1, arr_2, average='macro', zero_division='warn')
        accurecy_measures['precision'].append(precision)

        recall = recall_score(arr_1, arr_2, average='weighted')
        accurecy_measures['recall'].append(recall)

        counter = counter + 1
    return accurecy_measures

cisi_queries_result = matching_function(cisi_df.content,cisi_queries,cisi_answers_dictionary)

ranked_accurecy_headers = ['query_id','document_id','rank']
ranked_table = {'query_id':[],'document_id':[],'rank':[]}
for row in cisi_queries_result:
    temp_list = cisi_queries_result[row]
    i = 1
    for num in temp_list:
        ranked_table['query_id'].append(row)
        ranked_table['document_id'].append(num)
        ranked_table['rank'].append(i)
        i = i+1
ranked_df = pd.DataFrame(ranked_table)


answers_table = {'query_id':[],'document_id':[]}
for row in cisi_answers_dictionary:
    temp_list = cisi_answers_dictionary[row]
    for num in temp_list:
        answers_table['query_id'].append(row)
        answers_table['document_id'].append(num)
truths_df = pd.DataFrame(answers_table)

MAX_RANK = 1000

hits = pd.merge(truths_df, ranked_df,
    on=["query_id", "document_id"],
    how="left").fillna(MAX_RANK)

mrr = (1 / hits.groupby('query_id')['rank'].min()).mean()

cisi_queries_result = matching_function(cisi_df.content,cisi_queries,cisi_answers_dictionary)
cisi_accurecy_measures = calculate_measures(cisi_answers_dictionary.values() , cisi_queries_result.values(),1500)
cisi_accurecy_measures_df = pd.DataFrame(cisi_accurecy_measures)

precision_average = sum(cisi_accurecy_measures_df.precision) / len(cisi_accurecy_measures_df.precision)



cacm_queries_result = matching_function(cacm_df.content,cacm_queries,cacm_answers_dictionary)
cacm_accurecy_measures = calculate_measures(cacm_answers_dictionary.values() , cacm_queries_result.values(),3300)
cacm_accurecy_measures_df = pd.DataFrame(cacm_accurecy_measures)


precision_average = sum(cacm_accurecy_measures_df.precision) / len(cacm_accurecy_measures_df.precision)


from sklearn.feature_extraction.text import TfidfTransformer


# print(len(trainVectorizerArray))
# Documents

def query_documents(corpus_id, query_id):
    train_set = cisi_df.content
    documents = []
    answers_dictionary = cisi_answers_dictionary
    vectorizer = CountVectorizer(stop_words = stop_words)
    transformer = TfidfTransformer()
    new_test = []
    if (int(corpus_id) == 1):
        train_set = cisi_df.content
        answers_dictionary = cisi_answers_dictionary

    if (int(corpus_id) == 2):
        train_set = cacm_df.content
        answers_dictionary = cacm_answers_dictionary

    test = search_for_query(query_id, corpus_id)
    print(test)
    test = clean_text_round1(test)
    test = clean_text_round2(test)
    test = clean_text_round3(test)
    test = clean_text_round4(test)
    test = clean_text_round5(test)
    test = clean_text_round6(test)
    test = clean_text_round7(test)
    new_test.append(test)

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform(new_test).toarray()

    d_cosines = []
    for vector in trainVectorizerArray:
        for testV in testVectorizerArray:
            cosine = cx(vector, testV)
            d_cosines.append(cx(vector, testV))

    transformer.fit(trainVectorizerArray)
    number = len(answers_dictionary.get(int(query_id) + 1))
    out = np.array(d_cosines).argsort()[-number:][::-1]
    out.sort()
    i = 0

    bodys = []
    for num in out:
        out[i] = num + 1
        i = i + 1
    for num in out:
        documents.append(num)
        if (int(corpus_id) == 1):
            bodys.append(cisi_df.content.loc[num - 1])
        else:
            bodys.append(cacm_df.content.loc[num - 1])
        print(num)

    result = {'id': documents, 'body': bodys}


    return result


def search_for_query(query_id, queries_document_id):
    query_id = int(query_id)
    if (int(queries_document_id) == 1):
        return cisi_queries[query_id]
    else:
        return cacm_queries[query_id]






