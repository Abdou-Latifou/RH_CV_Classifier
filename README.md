CV Classifier App
Description
Cette application web intelligente utilise l'intelligence artificielle pour classifier automatiquement les CV (curricula vitae) en fonction de leur contenu textuel. Développée en Python avec Flask, elle offre une interface utilisateur moderne et intuitive pour analyser et catégoriser les documents de candidature selon différents profils professionnels.

Fonctionnalités Principales
Classification Automatique
Algorithmes Avancés : Utilise TF-IDF (Term Frequency-Inverse Document Frequency) combiné à la Régression Logistique pour analyser le contenu textuel des CV.
Précision Optimisée : Paramètres adaptatifs selon la taille du dataset pour garantir une précision réaliste de 80-85%.
Multi-Formats Supportés : Accepte les fichiers PDF, TXT, DOCX, PPTX, XLSX, ainsi que les images (avec OCR intégré).
Interface Web Moderne
Upload Drag & Drop : Interface conviviale permettant de glisser-déposer plusieurs CV simultanément.
Visualisation des Résultats : Affichage des profils prédits avec barres de confiance animées et probabilités détaillées.
Classification par Profil : Organisation automatique des CV classés dans des dossiers spécifiques.


Métriques de Performance
Rapport Détaillé : Page dédiée affichant l'accuracy, la matrice de confusion, et les métriques par profil.
Cross-Validation : Évaluation robuste avec validation croisée adaptée aux petits datasets.
Suivi des Métriques : Stockage persistant des performances du modèle.
Technologies Utilisées
Backend : Python 3.x, Flask
Machine Learning : Scikit-learn (TF-IDF, LogisticRegression, OneVsRestClassifier)
Traitement de Texte : NLTK (tokenization, stopwords, lemmatization)
Extraction de Texte : pdfplumber, python-docx, openpyxl, pptx, pytesseract (OCR)
Visualisation : Matplotlib, Seaborn
Frontend : HTML5, CSS3, JavaScript (Vanilla)
Installation et Utilisation

Prérequis : Python 3.8+, pip
Installation :
git clone https://github.com/votre-repo/cv-classifier-app.git
cd cv-classifier-app
pip install -r requirements.txt

Configuration : Modifier DATASET_PATH dans app1.py pour pointer vers votre dataset CSV.
Lancement :
python app1.py


Accéder à http://localhost:5001
Dataset Supporté
L'application détecte automatiquement le format des datasets :

Format Standard : Colonnes text et label
Format Resume.csv : Colonnes Resume_str et Category
Format Dataset.txt : Colonnes Text et Category
Architecture
CVClassifier Class : Gère l'extraction de texte, le préprocessing, l'entraînement et la prédiction.
Modèle Sauvegardé : Utilise pickle pour la persistance du modèle entraîné.
Gestion d'Erreurs : Traitement robuste des fichiers corrompus ou non supportés.
Contribution
Les contributions sont les bienvenues ! Veuillez :

Forker le projet
Créer une branche pour votre fonctionnalité
Commiter vos changements
Pousser vers la branche

Ouvrir une Pull Request
Licence
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
