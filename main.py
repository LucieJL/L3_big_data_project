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
#print(df['cps19_education'])
# Emploi : Remplacement de "je ne sais pas" par "autre"
df['cps19_employment']= df['cps19_employment'].replace({"Don't know/ Prefer not to answer": 'Other'})
#print(df['cps19_employment'])
# Religion : Groupement par "grand courant regligieux"
#Non religieux/oriental/juif/musulman/chretiens/chretiens déviré/other
df['cps19_religion']=df['cps19_religion'].replace({"None/ Don't have one/ Atheist":'Non religieux', 'Agnostic':'Non religieux', 'Buddhist/ Buddhism':'oriental', 'Hindu':'oriental', 'Jewish/ Judaism/ Jewish Orthodox':'juif', 'Muslim/ Islam': 'musulman', 'Sikh/ Sikhism':'oriental', "Anglican/ Church of England":'chretiens', 'Baptist':'chretiens', 'Catholic/ Roman Catholic/ RC':'chretiens', "Greek Orthodox/ Ukrainian Orthodox/ Russian Orthodox/ Eastern Orthodox": 'chretiens', "Jehovah's Witness":'chretiens', 'Lutheran':'chretiens', "Mormon/ Church of Jesus Christ of the Latter Day Saints":'chretiens déviré', 'Pentecostal/ Fundamentalist/ Born Again/ Evangelical':'chretiens déviré','Presbyterian':'chretiens','Protestant':'chretiens', 'United Church of Canada':'chretiens', 'Christian Reformed':'chretiens', 'Salvation Army':'chretiens', 'Mennonite':'chretiens déviré', 'Other (please specify)':'other',"Don't know/ Prefer not to answer":'other'})
#print(df['cps19_religion'])

#Genre rien à faire 

''' Séparation du dataset de test '''

dfTest = df[df.index.isin(testIndexes[0])]
df = df[~df.index.isin(testIndexes[0])]

''' Fusionner les étiquettes pour avoir une seule colonne label'''
data_label= df.iloc[:,22:31]
data_label = data_label.fillna('')
data_label['label'] = data_label['cps19_votechoice'] + data_label['cps19_votechoice_pr'] + data_label['cps19_vote_unlikely']+ data_label['cps19_vote_unlike_pr'] + data_label['cps19_v_advance']
print(data_label['label'])
''' Sélection des attributs '''

attributes = [
    'classeAge',
    'cps19_gender',
    'cps19_education',
    'cps19_employment',
    'cps19_religion',
]

df = df[attributes]
dfTest = dfTest[attributes]

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
