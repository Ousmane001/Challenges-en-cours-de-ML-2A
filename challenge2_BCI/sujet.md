VUE D'ENSEMBLE

Défi BCI

Qu'est-ce qu'une interface cerveau-ordinateur ?

Une interface cerveau-ordinateur (BCI) est un système qui permet à une personne d'interagir avec une machine sans aucune interaction physique. Il fonctionne en extrayant des caractéristiques des signaux neurophysiologiques (par exemple, les densités spectrales de puissance sur certaines bandes de fréquences) et en les attribuant à différentes classes. Ces classes peuvent être associées à des états cognitifs, des réponses sensorielles, etc., et les caractéristiques sont choisies de manière à être discriminatoires pour chaque classe. Nous appelons paradigme l'ensemble des tâches cognitives qu'un sujet est invité à effectuer lors de l'utilisation d'un système BCI ; différents paradigmes activent différents mécanismes cérébraux et produisent différentes caractéristiques de signal qui peuvent être utilisées plus tard comme caractéristiques pour la classification.

Dans cette compétition, vous travaillerez avec des données qui suivent le paradigme de l'imagerie motrice. Dans ce paradigme, on demande à un sujet d'imaginer un mouvement, par exemple en levant les mains, les pieds ou la langue lorsqu'un signal visuel est affiché sur un écran. Le fait d'imaginer volontairement un tel mouvement produit des ondes Mu dans le cortex moteur qui peuvent ensuite être identifiées par un algorithme de classification. La latéralité du mouvement imaginé (par exemple, le levage de la main gauche ou de la main droite) se reflète dans la latéralité de la production d'ondes Mu, avec différents modèles spatiaux EEG observés pour chaque classe de mouvement imaginé. Les systèmes BCI utilisant le paradigme de l'imagerie motrice (IM) remontent aux années 90 et sont encore souvent utilisés dans la pratique. Étant donné que les marqueurs discriminants de l'EEG enregistré sont liés aux oscillations dans la bande Mu, la plupart des classificateurs dans la littérature utilisent la densité spectrale de puissance des signaux dans chaque électrode comme caractéristiques.

Le défi

You are provided with data from six different subjects, each with a train and test partition. The data in X_train and X_test contain EEG signals collected from the scalp of each subject on several trials, while y_train informs the class of each trial, either left-hand or right-hand.

Votre objectif : construire un modèle prédictif en Python qui apprend la relation entre les signaux EEG et la tâche d'imagerie motrice pour chaque essai, et prédit avec précision les essais dans l'ensemble de tests (c'est-à-dire y_test).

Pourquoi est-ce important ?

Les BCI d'imagerie motrices ont des applications directes dans la technologie d'assistance et la réadaptation. Ils permettent aux personnes souffrant de handicaps moteurs graves - tels que celles causées par un accident vasculaire cérébral, la SLA ou une lésion de la moelle épinière - de contrôler les dispositifs externes (bras robotiques, fauteuils roulants, tableaux de communication) en utilisant uniquement leurs pensées. Au-delà de l'utilisation assistante, les MI-BCI sont également étudiés dans le contexte de la réadaptation motrice, où le neurofeedback peut aider les patients victimes d'un accident vasculaire cérébral à réentraîner leur cortex moteur en renforçant les modèles neuronaux associés à l'imagination du mouvement.

La classification précise de l'imagerie motrice est donc un problème central en neurosciences cliniques et appliquées. L'amélioration de la précision du décodage se traduit directement par des systèmes plus fiables et utilisables pour les vrais patients.

Approches suggérées

Il s'agit fondamentalement d'un problème de classification. Vous êtes encouragé à explorer diverses méthodes statistiques et d'apprentissage automatique, telles que :

Caractéristiques de puissance de bande + classificateurs linéaires : extraire la variance log ou la densité spectrale de puissance (PSD) dans les bandes Mu (8-12 Hz) et Beta (13-30 Hz) par électrode, puis appliquer l'analyse discriminante linéaire (LDA) ou la régression logistique.
Common Spatial Patterns (CSP) : une méthode BCI classique qui apprend les filtres spatiaux en maximisant le rapport de variance entre deux classes ; la puissance du signal filtré sert de caractéristiques pour un classificateur en aval.
Géométrie riemannienne : représenter chaque essai sous la forme d'une matrice de covariance et classer directement dans l'espace des matrices symétriques positives-définies à l'aide de méthodes riemanniennes basées sur la distance (par exemple, distance minimale à la moyenne, MDM).
Covariance régularisée + SVM : estimer les matrices de covariance de rétrécissement par essai et les alimenter (vectorisée) dans un noyau SVM.
Apprentissage profond : réseaux neuronaux convolutifs (par exemple, EEGNet, ShallowConvNet) qui apprennent les filtres spatiaux et temporaux de bout en bout à partir d'EEG brut ou filtré.
Notez que les données EEG sont très spécifiques au sujet. Les pipelines simples mais bien réglés par sujet surpassent souvent les modèles complexes qui ne tiennent pas compte de la variabilité inter-sujets.

Comment commencer

1. Télécharger les données

Une fois inscrit, téléchargez les données du concours à partir de ce lien. Vous devez télécharger et stocker ces fichiers dans un dossier data.

data/
    subject_A_X_train.npy
    subject_A_y_train.npy
    subject_A_X_test.npy
    subject_B_X_train.npy
    ...
    subject_F_X_test.npy
2. Charger les données

import numpy as np

X_train = np.load("data/subject_A_X_train.npy")  # shape: (140, 64, 1537)
y_train = np.load("data/subject_A_y_train.npy")  # shape: (140,)
X_test  = np.load("data/subject_A_X_test.npy")   # shape: (60, 64, 1537)
Each trial is a matrix of shape (64 channels, 1537 time points). Labels are strings: 'left_hand' or 'right_hand'.

3. Extraire des caractéristiques et construire un modèle

Les essais EEG bruts ne peuvent pas être alimentés directement à la plupart des classificateurs - vous devez d'abord extraire les caractéristiques. Une ligne de référence simple et efficace utilise la variance log de chaque canal dans le temps, suivie de l'analyse discriminante linéaire (LDA) :

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def temporal_variance_log(X):
    # X shape: (n_epochs, 64, T) -> output: (n_epochs, 64)
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


pipeline = Pipeline([
    ('log_var', FunctionTransformer(temporal_variance_log)),
    ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])

pipeline.fit(X_train, y_train)
4. Générer des prédictions

import pandas as pd

y_pred = pipeline.predict(X_test)
pd.DataFrame({'y_pred': y_pred}).to_csv("subject_A_y_pred.csv", index=False)
The output file subject_A_y_pred.csv should look like:

y_pred
left_hand
right_hand
left_hand
...
Chaque fichier doit contenir exactement 60 lignes (à l'exclusion de l'en-tête).

5. soumettre

Create one prediction file per subject (subject_A_y_pred.csv through subject_F_y_pred.csv), put them all in a single zip file, and upload it on the competition platform (see the Evaluation page for details).

Résumé

Le pipeline de travail minimal (pour tous les sujets) est :

import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def temporal_variance_log(X):
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


pipeline = Pipeline([
    ('log_var', FunctionTransformer(temporal_variance_log)),
    ('clf', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
])

for subject in ['A', 'B', 'C', 'D', 'E', 'F']:
    X_train = np.load(f'data/subject_{subject}_X_train.npy')
    y_train = np.load(f'data/subject_{subject}_y_train.npy')
    X_test  = np.load(f'data/subject_{subject}_X_test.npy')

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    pd.DataFrame({'y_pred': y_pred}).to_csv(f'subject_{subject}_y_pred.csv', index=False)
Remplacez l'extracteur de fonctionnalités ou le classificateur par votre propre approche et itérez à partir de là !


Évaluation

Métrique

Les soumissions sont évaluées à l'aide de la précision de la classification - la proportion d'essais de test correctement classés :

Précision
=
1
N
∑
Je
=
1
N
1
[
Y
^
Je
=
Y
Je
]
Précision= 
N
1
  
Je = 1
∑
N
 1[ 
Y
^
  
Je
 =Y 
Je
 ]

Où
Y
^
Je
Y
^
  
Je
 C'est la classe prévue et
Y
Je
Y 
Je
 est la vraie classe pour le procès
Je
Je.

Le score final est la précision moyenne sur les six sujets :

Score
=
1
6
∑
S
∈
{
A
,
B
,
C
,
D
,
E
,
F
}
Précision
S
Score= 
6
1
  
S∈{A,B,C,D,E,F}
∑
 Précision 
S
 

Un score plus élevé est meilleur. Le classement est classé par ordre décroissant de précision moyenne.

Format de soumission

You must submit one CSV file per subject, named subject_A_y_pred.csv through subject_F_y_pred.csv. Each file must contain a single column y_pred with the predicted class for each trial in X_test, in the same order.

Format attendu (exemple pour le sujet A) :

y_pred
left_hand
right_hand
left_hand
...
Chaque fichier doit contenir exactement 60 lignes (à l'exclusion de l'en-tête), correspondant aux 60 essais de test.
Each value must be either left_hand or right_hand.
La colonne doit être nommée y_pred.
All prediction files subject_A_y_pred.csv through subject_F_y_pred.csv should be put into a single zip file that will be submitted to the Codabench platform.

Comment fonctionne la notation

Pour chaque sujet, la précision est calculée, puis moyenne sur les sujets :

import numpy as np
import pandas as pd

subjects = ['A', 'B', 'C', 'D', 'E', 'F']
accuracies = []

for subject in subjects:
    y_pred = pd.read_csv(f'subject_{subject}_y_pred.csv')['y_pred'].values
    y_test = np.load(f'subject_{subject}_y_test.npy')
    accuracies.append(np.mean(y_pred == y_test))

score = np.mean(accuracies)




Données

Fichiers

The challenge data is provided as NumPy (.npy) files, one set per subject. Each subject S has three files:

Fichier	Forme	description
subject_S_X_train.npy	(140, 64, 1537)	Formation aux essais EEG
subject_S_y_train.npy	(140,)	Étiquettes de formation
subject_S_X_test.npy	(60, 64, 1537)	Essais de test EEG
Les fichiers peuvent être chargés par exemple avec

import numpy as np
data = {}
for subject in ['A', 'B', 'C', 'D', 'E', 'F']:
  data[f'subject_{subject}'] = {}
  data[f'subject_{subject}']['X_train'] = np.load(f'subject_{subject}_X_train.npy')  # shape: (140, 64, 1537)
  data[f'subject_{subject}']['y_train'] = np.load(f'subject_{subject}_y_train.npy')  # shape: (140,)
  data[f'subject_{subject}']['X_test'] = np.load(f'subject_{subject}_X_test.npy')  # shape: (60, 64, 1537)
Essais EEG (X_train, X_test)

Chaque tableau a trois dimensions : (essais, canaux, points de temps).

Essais - 140 essais de formation et 60 essais de test par sujet.
Canaux - 64 électrodes EEG placées sur le cuir chevelu.
Points de temps - 1537 échantillons par essai (dtype : float64.
Étiquettes (y_train)

Chaque entrée est une chaîne indiquant la classe d'imagerie motrice effectuée au cours de cet essai :

'left_hand'— le sujet a imaginé un mouvement de la main gauche
'right_hand'- le sujet a imaginé un mouvement de la main droite
Les deux classes sont équilibrées (70 essais chacune par sujet).

Caractéristiques clés

Données par sujet : chaque sujet a sa propre division train/test. Les modèles doivent être formés et évalués indépendamment par sujet.
Entrée 3D : contrairement aux données tabulaires, chaque essai est une matrice de forme (64, 1537). La plupart des classificateurs nécessitent l'extraction de caractéristiques (par exemple, puissance de bande, covariance) avant l'ajustement.
Aucune valeur manquante : tous les tableaux sont entièrement remplis.




