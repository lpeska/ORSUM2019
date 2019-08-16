import random
import numpy as np
import pandas as pd

from numpy.random import beta


# methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
# methodsParamsDF:pd.DataFrame[numberOfVotes:int], topK:int
def aggrElectionsRun(methodsResultDict, methodsParamsDF, topK = 20):

  if sorted([mI for mI in methodsParamsDF.index]) != sorted([mI for mI in methodsResultDict.keys()]):
    raise ValueError("Arguments methodsResultDict and methodsParamsDF have to define the same methods.")

  if np.prod([len(methodsResultDict.get(mI)) for mI in methodsResultDict]) == 0:
    raise ValueError("Argument methodsParamsDF contains in ome method an empty list of items.")

  if topK < 0 :
    raise ValueError("Argument topK must be positive value.")

  candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
  uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))
  #print("UniqueCandidatesI: ", uniqueCandidatesI)

  # numbers of elected candidates of parties
  electedOfPartyDictI = {mI:1 for mI in methodsParamsDF.index}
  #print("ElectedForPartyI: ", electedOfPartyDictI)

  # votes number of parties
  votesOfPartiesDictI = {mI:methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}
  #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

  recommendedItemIDs = []

  for iIndex in range(0, topK):
    #print("iIndex: ", iIndex)

    if len(uniqueCandidatesI) == 0:
        return recommendedItemIDs[:topK]

    # coumputing of votes of remaining candidates
    actVotesOfCandidatesDictI = {}
    for candidateIDJ in uniqueCandidatesI:
       votesOfCandidateJ = 0
       for parityIDK in methodsParamsDF.index:
          partyAffiliationOfCandidateKJ = methodsResultDict[parityIDK].get(candidateIDJ, 0)
          votesOfPartyK = votesOfPartiesDictI.get(parityIDK)
          votesOfCandidateJ += partyAffiliationOfCandidateKJ * votesOfPartyK
       actVotesOfCandidatesDictI[candidateIDJ] = votesOfCandidateJ
    #print(actVotesOfCandidatesDictI)

    # get the highest number of votes of remaining candidates
    maxVotes = max(actVotesOfCandidatesDictI.values())
    #print("MaxVotes: ", maxVotes)

    # select candidate with highest number of votes
    selectedCandidateI = [votOfCandI for votOfCandI in actVotesOfCandidatesDictI.keys() if actVotesOfCandidatesDictI[votOfCandI] == maxVotes][0]
    #print("SelectedCandidateI: ", selectedCandidateI)

    # add new selected candidate in results
    recommendedItemIDs.append(selectedCandidateI);

    # removing elected candidate from list of candidates
    uniqueCandidatesI.remove(selectedCandidateI)

    # updating number of elected candidates of parties
    electedOfPartyDictI = {partyIDI:electedOfPartyDictI[partyIDI] + methodsResultDict[partyIDI].get(selectedCandidateI, 0) for partyIDI in electedOfPartyDictI.keys()}
    #print("DevotionOfPartyDictI: ", devotionOfPartyDictI)

    # updating number of votes of parties
    votesOfPartiesDictI = {partyI:methodsParamsDF.votes.loc[partyI] / electedOfPartyDictI.get(partyI)  for partyI in methodsParamsDF.index}
    #print("VotesOfPartiesDictI: ", votesOfPartiesDictI)

  return recommendedItemIDs[:topK]


# methodsResultDict:{String:pd.Series(rating:float[], itemID:int[])},
# methodsParamsDF:pd.DataFrame[numberOfVotes:int], topK:int
def aggrElectionsRunWithResponsibility(methodsResultDict, methodsParamsDF, topK = 20):
  
  # recommendedItemIDs:int[]
  recommendedItemIDs = aggrElectionsRun(methodsResultDict, methodsParamsDF, topK)
  votesOfPartiesDictI = {mI:methodsParamsDF.votes.loc[mI] for mI in methodsParamsDF.index}
  
  candidatesOfMethods = np.asarray([cI.keys() for cI in methodsResultDict.values()])
  uniqueCandidatesI = list(set(np.concatenate(candidatesOfMethods)))

  candidateOfdevotionOfPartiesDictDict = {}
  for candidateIDI in recommendedItemIDs:
  #for candidateIDI in uniqueCandidatesI:
     devotionOfParitiesDict = {}
     for parityIDJ in methodsParamsDF.index:
        devotionOfParitiesDict[parityIDJ] = methodsResultDict[parityIDJ].get(candidateIDI, 0)  * votesOfPartiesDictI[parityIDJ]
     candidateOfdevotionOfPartiesDictDict[candidateIDI] = devotionOfParitiesDict
  #print(candidateOfdevotionOfPartiesDictDict)

  # selectedCandidate:[itemID:{methodID:responsibility,...},...]
  selectedCandidate = [(candidateI, candidateOfdevotionOfPartiesDictDict[candidateI]) for candidateI in recommendedItemIDs]

  return selectedCandidate



if __name__== "__main__":
  print("Running Elections:")

  # number of recommended items
  N = 120

  # method results, items=[1,2,4,5,6,7,8,12,32,64,77]
  methodsResultDict = {
          "metoda1":pd.Series([0.2,0.1,0.3,0.3,0.1],[32,2,8,1,4],name="rating"),
          "metoda2":pd.Series([0.1,0.1,0.2,0.3,0.3],[1,5,32,6,7],name="rating"),
          "metoda3":pd.Series([0.3,0.1,0.2,0.3,0.1],[7,2,77,64,12],name="rating")
          }
  #print(methodsResultDict)


  # methods parametes
  methodsParamsData = [['metoda1',100], ['metoda2',80], ['metoda3',60]]
  methodsParamsDF = pd.DataFrame(methodsParamsData, columns=["methodID","votes"])
  methodsParamsDF.set_index("methodID", inplace=True)

  #print(methodsParamsDF)


  #itemIDs = aggrElectionsRun(methodsResultDict, methodsParamsDF, N)
  itemIDs = aggrElectionsRunWithResponsibility(methodsResultDict, methodsParamsDF, N)

  print(itemIDs)

