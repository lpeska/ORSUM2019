# ORSUM2019
### Repository for paper "Fuzzy D'Hondt's Algorithm for On-line Recommendations Aggregation" by Ladislav Peska and Stepan Balcar, presented at ORSUM 2019 workshop of the RecSys conference.

## Abstract
In this paper, we present Fuzzy D'Hondt's algorithm suitable to aggregate lists of recommended objects originating from various base recommending methods. The algorithm is inspired by D'Hondt's election method used to a proportional conversion of votes to mandates in public elections. We enhance the original approach to enable fuzzy candidate-party membership, propose a gradient learning of per-party votes assignments and utilize it for iterative on-line aggregation of recommendations. Main features of the proposed algorithm are ability to iteratively learn relevance of individual base recommenders (parties), ability to account for multiple item's memberships and capability to provide proportional representation of base recommenders w.r.t. their results as well as fair ordering of the final list of recommended items. Fuzzy D'Hondt's aggregation method was evaluated in on-line A/B testing against state-of-the-art approach based on multi-armed bandits with Thompson sampling and achieved competitive results. 

## Repository content
Source codes stored in this repository represents a simple RS server utilized for the evaluation of Fuzzy D'Hondt's algorithm (evaluation was performed on a Czech travel agency). The server contains several base recommending algorithms (variants of doc2vec, word2vec and cosine CB) as well as Fuzzy D'Hondt's and Multiarmed Bandit algorithms for aggregation of base recommendations. The server assumes to collect implicit feedback from the travel agency's users (based on which, the parameters of aggregation algorithms are updated) and upon request, it returns a list of recommended items w.r.t. selected RS.

## Fuzzy D'Hondt's algorithm
- Fuzzy D'Hondt's selection algorithm is implemented in aggr_elections.py 
- Iterative votes updating algorithm is implemented in update_dhondt_params() method of the server.py file

## Dataset
- The dataset utilized in this paper comes from a mid-sized Czech travel agency. The data.zip file contains all dataset tables (CB attributes, textual description and user's feedback). 
- Note that due to the source of the data, some information are written in Czech (e.g., tour descriptions and CB attribute names & values)
- Dataset pre-processing steps are defined in OfflineDataPreparation.ipynb, training base recommenders is implemented in createOnlineModels.py file


