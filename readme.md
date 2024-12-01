
# Analyse et Segmentation Client avec Python

---

## **Objectif**

Ce projet vise à explorer, nettoyer, analyser et segmenter un dataset client à l'aide de Python. Vous apprendrez à :
1. Explorer et préparer les données.
2. Réaliser une analyse exploratoire (EDA) et segmenter les clients en groupes.
3. Effectuer une analyse en composantes principales (PCA) pour simplifier les données.
4. Formuler des recommandations stratégiques basées sur les résultats.

---

## **Configuration et Installation**

### **Étape 1 : Installation de Python**

#### **1. Vérifiez si Python est déjà installé sur votre machine**
- Ouvrez un terminal (macOS/Linux) ou PowerShell (Windows) et exécutez :
  ```bash
  python --version
  ```
  Si Python est installé, vous verrez une version (par exemple, `Python 3.9.7`).
  
- Si Python n'est pas installé, passez à l'étape suivante.

#### **2. Installer Python sur votre machine**

**Pour Windows :**
1. Téléchargez Python depuis le site officiel : [https://www.python.org/downloads/](https://www.python.org/downloads/).
2. Lancez l'installateur et **cochez la case "Add Python to PATH"** avant d'installer.
3. Une fois l'installation terminée, ouvrez PowerShell et tapez :
   ```bash
   python --version
   ```
   Cela doit afficher la version de Python.

**Pour macOS :**
1. Ouvrez un terminal et installez Homebrew (si ce n’est pas déjà fait) :
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Installez Python avec Homebrew :
   ```bash
   brew install python
   ```
3. Vérifiez l'installation :
   ```bash
   python3 --version
   ```

**Pour Linux :**
1. Installez Python via le gestionnaire de paquets :
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip
   ```
2. Vérifiez l'installation :
   ```bash
   python3 --version
   ```

---

### **Étape 2 : Installation des Dépendances**

#### **1. Créez un environnement virtuel (optionnel, mais recommandé)**
Un environnement virtuel permet d’isoler les dépendances du projet.

1. **Créez l’environnement virtuel :**
   ```bash
   python -m venv env
   ```

2. **Activez l’environnement :**
   - **Sous Windows :**
     ```bash
     .\env\Scripts\activate
     ```
   - **Sous macOS/Linux :**
     ```bash
     source env/bin/activate
     ```

#### **2. Installez les dépendances requises**
Dans le même terminal, exécutez la commande suivante pour installer toutes les bibliothèques nécessaires :
```bash
pip install pandas openpyxl matplotlib seaborn scikit-learn
```

- **Dépendances installées :**
  - `pandas` : Manipulation et analyse de données.
  - `openpyxl` : Lecture des fichiers Excel (`.xlsx`).
  - `matplotlib` : Création de graphiques.
  - `seaborn` : Visualisation avancée de données.
  - `scikit-learn` : Outils de machine learning pour la segmentation.


#### **3. Testez les installations**
Exécutez cette commande pour vérifier que toutes les bibliothèques sont installées :
```bash
pip list
```
Vous devriez voir les bibliothèques listées avec leurs versions.

---

## **Plan de Travail**

Le projet est divisé en trois jours :

---

### **Jour 1 : Exploration et Préparation des Données**  
Durée estimée : 4 heures

1. **Chargement des bibliothèques nécessaires** :
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.preprocessing import StandardScaler
   ```

2. **Prévisualisation des données** :
   - Charger les données et afficher un aperçu :
     ```python
     df = pd.read_excel("Camp_Market 1.xlsx", engine="openpyxl")
     print(df.head())
     print(df.info())
     ```

3. **Nettoyage et préparation des données** :
   - Remplir les valeurs manquantes :
     ```python
     df.fillna(df.mean(), inplace=True)
     ```
   - Supprimer les colonnes inutiles :
     ```python
     df.drop(columns=["Z_CostContact", "Z_Revenue"], errors="ignore", inplace=True)
     ```
   - Supprimer les doublons :
     ```python
     df.drop_duplicates(inplace=True)
     ```

4. **Exporter les données nettoyées** :
   - Sauvegarder les données nettoyées :
     ```python
     df.to_csv("cleaned_data.csv", index=False)
     ```

---

### **Jour 2 : Analyse Exploratoire et Segmentation Client (Clustering)**  
Durée estimée : 6 heures

1. **Analyse exploratoire des données (EDA)** :
   - Histogrammes :
     ```python
     df.hist(bins=30, figsize=(15, 10))
     plt.tight_layout()
     plt.show()
     ```
   - Heatmap pour les corrélations :
     ```python
     corr_matrix = df.corr()
     plt.figure(figsize=(10, 8))
     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
     plt.show()
     ```

2. **Analyse en composantes principales (PCA)** :
   - Standardiser les colonnes pertinentes :
     ```python
     scaler = StandardScaler()
     data_scaled = scaler.fit_transform(df[["MntWines", "MntFruits", "MntMeatProducts", "NumWebVisitsMonth"]])
     ```
   - Réaliser une PCA :
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=3)
     pca_data = pca.fit_transform(data_scaled)
     print("Variance expliquée :", pca.explained_variance_ratio_)
     ```

3. **Segmentation des clients (Clustering)** :
   - Déterminer le nombre optimal de clusters avec l'Elbow Method :
     ```python
     from sklearn.cluster import KMeans
     inertias = []
     for k in range(1, 10):
         kmeans = KMeans(n_clusters=k, random_state=42)
         kmeans.fit(pca_data)
         inertias.append(kmeans.inertia_)
     plt.plot(range(1, 10), inertias, marker='o')
     plt.show()
     ```
   - Appliquer K-Means pour segmenter les clients :
     ```python
     kmeans = KMeans(n_clusters=3, random_state=42)
     df["Cluster"] = kmeans.fit_predict(pca_data)
     ```

4. **Visualisation des clusters** :
   - Scatter plot des clusters :
     ```python
     sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df["Cluster"], palette="viridis")
     plt.show()
     ```

---

### **Jour 3 : Modélisation Prédictive et Recommandations**  
Durée estimée : 5 heures

1. **Modélisation prédictive (Régression Logistique)** :
   - Prédire la réponse (`Response`) :
     ```python
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LogisticRegression

     X = df.drop(columns=["Response", "Cluster"], errors="ignore")
     y = df["Response"]
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     X_train = scaler.fit_transform(X_train)
     X_test = scaler.transform(X_test)

     logreg = LogisticRegression()
     logreg.fit(X_train, y_train)
     y_pred = logreg.predict(X_test)
     ```

2. **Interprétation et recommandations** :
   - Identifier les clusters à fort potentiel pour des offres spécifiques.
   - Visualiser les résultats via des graphiques.

---

## **Fichiers Fournis**

- `Camp_Market 1.xlsx` : Données clients.
- `main.py` : Script Python principal.
- `cleaned_data.csv` et `segmented_data.csv` : Fichiers générés.

---
