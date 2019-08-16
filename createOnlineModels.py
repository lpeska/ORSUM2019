import numpy as np
import pandas as pd
import doc2vec
import word2vec
import rank_metrics
import datetime
import pickle
from collections import defaultdict, OrderedDict
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval

#listOfAlgs = [ "attributeCosineSim", "doc2vec", "word2vec"]
listOfAlgs = [  "doc2vec"]
embedSizes = [32, 64, 128]
windowSizes = [1, 3, 5]


dfValidDates = pd.read_csv("data/serialValidDates.csv", sep=";", header=0, index_col=0)
dfValidDates.novelty_date = pd.to_datetime(dfValidDates.novelty_date)
now = datetime.datetime.now()
novelty_score = 1 / np.log((now - dfValidDates.novelty_date).dt.days + 2.72)
#print(novelty_score)
dfValidDates["noveltyScore"] = novelty_score
dct = defaultdict(int)
noveltyDict = dfValidDates.noveltyScore.to_dict(into=dct)

dfCBFeatures = pd.read_csv("data/serialCBFeatures.txt", sep=",", header=0, index_col=0)
dfCBSim = 1 - pairwise_distances(dfCBFeatures, metric="cosine")


dfCBSimNoSame = np.copy(dfCBSim)
np.fill_diagonal(dfCBSimNoSame, 0.0)

cbNames = dfCBFeatures.index.values
cbVals = range(len(cbNames))
rev_cbDict = dict(zip(cbVals, cbNames))
cbDict = dict(zip(cbNames, cbVals))

print(dfCBSim.shape)
print(dfCBSim[0:5,0:5])

df = pd.read_csv("data/serialTexts.txt", sep=";", header=0, index_col=0)
d2v_names = df.index.values
d2v_vals = range(len(d2v_names))

rev_dict_d2v = dict(zip(d2v_vals, d2v_names))
dict_d2v = dict(zip(d2v_names, d2v_vals))



#print(dict_d2v)
#print(rev_dict_d2v)
print(len(d2v_names), len(d2v_vals), len(dict_d2v), len(rev_dict_d2v))

#For off-line evaluation only
#testSet = pd.read_csv("data/test_data_wIndex.txt", sep=",", header=0, index_col=0)
#testSet["oids"] = testSet.strOID.str.split()
#trainSet = pd.read_csv("data/train_data_wIndex.txt", sep=",", header=0, index_col=0)
#trainSet["oids"] = trainSet.strOID.str.split()

trainTimeWeight = pd.read_csv("data/serialLogDays.txt", sep=",", header=0, index_col=0, converters={"logDays": literal_eval})
print(trainTimeWeight.head())
#trainTimeWeight["weights"] = trainTimeWeight.logDays.str.split()

def mmr_objects_similarity(i, j, rev_dict):
    try:
        idi = cbDict[rev_dict[i]]
        idj = cbDict[rev_dict[j]]
        return dfCBSim[idi, idj]
    except:
        return 0

def mmr_sorted(docs, lambda_, results, rev_dict, length):
    """Sort a list of docs by Maximal marginal relevance

	Performs maximal marginal relevance sorting on a set of
	documents as described by Carbonell and Goldstein (1998)
	in their paper "The Use of MMR, Diversity-Based Reranking
	for Reordering Documents and Producing Summaries"

    :param docs: a set of documents to be ranked
				  by maximal marginal relevance
    :param q: query to which the documents are results
    :param lambda_: lambda parameter, a float between 0 and 1
    :return: a (document, mmr score) ordered dictionary of the docs
			given in the first argument, ordered my MMR
    """
    #print("enter to MMR")
    selected = OrderedDict()
    docs = set(docs)
    while (len(selected) < len(docs)) and (len(selected) < length):
        remaining = docs - set(selected)
        mmr_score = lambda x: lambda_ * results[x] - (1 - lambda_) * max(
            [mmr_objects_similarity(x, y, rev_dict) for y in set(selected) - {x}] or [0])
        next_selected = argmax(remaining, mmr_score)
        selected[next_selected] = len(selected)
        #print(len(selected))
    return selected


def argmax(keys, f):
    return max(keys, key=f)

def user_novelty_at_n(rankedIDs, trainModelIDs, n):
    return np.sum([1 for i in rankedIDs[0:n] if i in trainModelIDs])/n

def prec_at_n(rankedRelevance, n):
    return np.sum(rankedRelevance[0:n])/n

def meanNovelty_at_n(noveltyList, n):
    return np.sum(noveltyList[0:n])/n

def rec_at_n(rankedRelevance, n):
    return np.sum(rankedRelevance[0:n])/np.sum(rankedRelevance)

def ild_at_n(idx, rev_dict,  n):
    divList = []
    for i in idx[0:n]:
        for j in idx[0:n]:
            try:
                idi = cbDict[rev_dict[i]]
                idj = cbDict[rev_dict[j]]
                if i != j:
                    divList.append(1-dfCBSim[idi, idj])
            except:
                pass
    return np.mean(divList)


def save_obj(obj, name):
    with open('objAll/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open('objAll/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


with open("resultsWithNovDiv_32_0dot01Temporal.csv","w") as resultsFile:
    resultsFile.write("uid;alg;params;recAlg;noveltyEnhance;diversityEnhance;r2Score;mae;map;aucScore;mrr;p5;p10;r5;r10;nDCG10;nDCG100;nDCGFull;novelty5;novelty10;user_novelty5;user_novelty10;ild5;ild10\n")
    for alg in listOfAlgs:
        if alg == "word2vec":
            for e in embedSizes:
                for w in windowSizes:
                    model, rev_dict, dictionary = word2vec.word2vecRun(w,e)
                    dictionary = dict([((int(i),j) if i !="RARE" else (-1,j)) for i,j in dictionary.items() ])
                    rev_dict = dict(zip(dictionary.values(), dictionary.keys()))
                    #store models

                    save_obj(model, "word2vec_{0}_{1}_model".format(e,w))
                    save_obj(dictionary, "word2vec_{0}_{1}_dict".format(e, w))
                    save_obj(rev_dict, "word2vec_{0}_{1}_revdict".format(e, w))

                    #print("eval W2V")
                    #eval(model, dictionary, rev_dict, testSet, trainSet, alg, (e, w), resultsFile)
        elif alg == "doc2vec":
            for e in embedSizes:
                for w in windowSizes:
                    model = doc2vec.doc2vecRun(w,e)
                    rev_dict = rev_dict_d2v
                    dictionary = dict_d2v
                    # store models

                    save_obj(model, "doc2vec_{0}_{1}_model".format(e, w))
                    save_obj(dictionary, "doc2vec_dict")
                    save_obj(rev_dict, "doc2vec_revdict")

                    #print("eval D2V")
                    #eval(model, dictionary, rev_dict, testSet, trainSet, alg, (e,w), resultsFile)
        else:

            rev_dict = rev_cbDict
            dictionary = cbDict

            save_obj(dictionary, "vsm_dict")
            save_obj(rev_dict, "vsm_revdict")

            for same in ["sameAllowed", "noSameObjects"]:
                if same == "sameAllowed":
                    model = dfCBSim
                    save_obj(model, "vsm_{0}_model".format(same))
                else:
                    model = dfCBSimNoSame
                    save_obj(model, "vsm_{0}_model".format(same))
                #print("eval CB")
                #eval(model, dictionary, rev_dict, testSet, trainSet, alg, [same], resultsFile)
