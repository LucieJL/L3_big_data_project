import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

''' Paramétrage '''

df = pd.read_csv('CES19.csv')
testIndexes = pd.read_csv('exemple.txt', sep='\t', header=None)

''' Pipeline de prétraitrement '''

# Age : Division des classes d'âges existantes en fonction de leur médiane
col = "cps19_age"
choices = [1,2,3,4,5,6]
conditions = [(df[col]>=18) & (df[col]<29),
            (df[col]>=29) & (df[col]<35),
            (df[col]>=35) & (df[col]<45),
            (df[col]>=45) & (df[col]<55),
            (df[col]>=55) & (df[col]<66),
            (df[col]>=66)]
df["classeAge"] = np.select(conditions, choices, default=0)


# Education : Numérisation/Ordonnement
df['cps19_education']= df['cps19_education'].replace({'No schooling':0, 'Some elementary school':1, 'Completed elementary school':2,'Some secondary/ high school': 3, 'Completed secondary/ high school': 4, 'Some technical, community college, CEGEP, College Classique': 5, 'Completed technical, community college, CEGEP, College Classique': 6, 'Some university': 7, "Bachelor's degree": 8, "Master's degree":9, 'Professional degree or doctorate': 10, "Don't know/ Prefer not to answer": -1})
print(df['cps19_education'])
# Emploi : Remplacement de "je ne sais pas" par "autre"
df['cps19_employment']= df['cps19_employment'].replace({"Don't know/ Prefer not to answer": 'Other'})
print(df['cps19_employment'])
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
