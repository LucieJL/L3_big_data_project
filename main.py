import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

''' Paramétrage '''

df = pd.read_csv('CES19.csv', dtype=str)
testIndexes = pd.read_csv('exemple.txt', sep='\t', header=None)

''' Pipeline de prétraitrement '''

# Age : Division des classes d'âges existantes en fonction de leur médiane
# Education : Numérisation/Ordonnement
# Emploi : Remplacement de "je ne sais pas" par "autre"
# Religion : Groupement par "grand courant regligieux"

''' Séparation du dataset de test '''

dfTest = df[df.index.isin(testIndexes[0])]
df = df[~df.index.isin(testIndexes[0])]

''' Entrainement '''

# Utilisation de "k-fold cross validation"
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(df):
    print("TRAIN:", train_index, "TEST:", test_index)

    # Classificateur naïf de bayes
    # https://www.stat4decision.com/fr/foret-aleatoire-avec-python/

    # Arbre de décision (Random Forest)
    # https://www.cours-gratuit.com/tutoriel-python/tutoriel-python-matriser-la-classification-nave-baysienne-avec-scikit-learn

    # K plus proches voisins
    # https://medium.com/@kenzaharifi/bien-comprendre-lalgorithme-des-k-plus-proches-voisins-fonctionnement-et-impl%C3%A9mentation-sur-r-et-a66d2d372679

''' Test de l'algorithme '''

# Classificateur naïf de bayes
# Arbre de décision (Random Forest)
# K plus proches voisins
