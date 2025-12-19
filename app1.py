import re
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from threading import Thread
from classifier import CVClassifier
from email_processor import start_email_monitoring
import shutil



# Télécharger TOUS les modèles NLTK nécessaires (une fois pour toutes)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = 'super_secret_key_2025'

# Dossiers des profils classifiés directement à la racine pour accès direct
CLASSIFIED_ROOT = 'downloads'  # Dossiers comme downloads/Developpeur, downloads/Manager, etc.

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(CLASSIFIED_ROOT, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'pptx', 'xlsx', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
METADATA_FILE = 'cv_metadata.json'


classifier = CVClassifier()

# Chemin vers votre dataset - MODIFIEZ CE CHEMIN
DATASET_PATH = r"C:\Users\akedjeri\Desktop\cv_classifier_app\sample_dataset.csv"

if os.path.exists('cv_classifier_model.pkl'):
    classifier.load_model()
    print("Modèle chargé")
else:
    if os.path.exists(DATASET_PATH):
        print("Chargement du dataset...")
        texts, labels = classifier.load_dataset(DATASET_PATH)
        print(f"Dataset chargé: {len(texts)} échantillons")
        classifier.train(texts, labels)
        classifier.save_model()
        print("Modèle entraîné et sauvegardé")
    else:
        print(f"Dataset non trouvé à {DATASET_PATH}. Veuillez spécifier le bon chemin.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_metadata(metadata):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def add_cv_metadata(profile, filename):
    metadata = load_metadata()
    cv_key = f"{profile}/{filename}"
    metadata[cv_key] = {
        'profile': profile,
        'filename': filename,
        'classified_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'active'
    }
    save_metadata(metadata)




@app.route('/')
def index():
    return render_template('index.html')

# Route pour obtenir la liste des dossiers et leur contenu
@app.route('/get_folders')
def get_folders():
    folders = []
    metadata = load_metadata()

    try:
        if os.path.exists(CLASSIFIED_ROOT):
            for folder_name in os.listdir(CLASSIFIED_ROOT):
                folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
                if os.path.isdir(folder_path):
                    files_with_dates = []
                    for f in os.listdir(folder_path):
                        if os.path.isfile(os.path.join(folder_path, f)):
                            cv_key = f"{folder_name}/{f}"
                            # Récupérer la date depuis les métadonnées ou la date de modification du fichier
                            if cv_key in metadata and 'classified_date' in metadata[cv_key]:
                                classified_date = metadata[cv_key]['classified_date']
                            else:
                                # Fallback: utiliser la date de modification du fichier
                                file_mtime = os.path.getmtime(os.path.join(folder_path, f))
                                classified_date = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')

                            files_with_dates.append({
                                'filename': f,
                                'classified_date': classified_date
                            })

                    # Trier par date décroissante (plus récent en premier)
                    files_with_dates.sort(key=lambda x: x['classified_date'], reverse=True)

                    # Trouver la date du CV le plus récent pour ce dossier
                    latest_date = files_with_dates[0]['classified_date'] if files_with_dates else None

                    folders.append({
                        'name': folder_name,
                        'count': len(files_with_dates),
                        'files': files_with_dates,
                        'latest_date': latest_date
                    })

            # Trier les dossiers par date du CV le plus récent
            folders.sort(key=lambda x: x['latest_date'] or '', reverse=True)

    except Exception as e:
        print(f"Erreur lors de la récupération des dossiers: {e}")

    return jsonify({'folders': folders})

# Route pour télécharger un fichier spécifique d'un dossier
@app.route('/download_folder/<path:folder_name>')
def download_folder(folder_name):
    try:
        # Ne pas utiliser secure_filename ici car il supprime les underscores
        folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
        metadata = load_metadata()

        print(f"Tentative d'accès au dossier: {folder_path}")  # Debug

        if not os.path.exists(folder_path):
            return f"Dossier non trouvé: {folder_path}", 404

        if not os.path.isdir(folder_path):
            return "Ce n'est pas un dossier valide", 400

        # Récupérer les fichiers avec leurs dates
        files_with_dates = []
        for f in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, f)):
                cv_key = f"{folder_name}/{f}"
                # Récupérer la date depuis les métadonnées ou la date de modification du fichier
                if cv_key in metadata and 'classified_date' in metadata[cv_key]:
                    classified_date = metadata[cv_key]['classified_date']
                else:
                    # Fallback: utiliser la date de modification du fichier
                    file_mtime = os.path.getmtime(os.path.join(folder_path, f))
                    classified_date = datetime.fromtimestamp(file_mtime).strftime('%Y-%m-%d %H:%M:%S')

                files_with_dates.append({
                    'filename': f,
                    'classified_date': classified_date
                })

        # Trier par date décroissante (plus récent en premier)
        files_with_dates.sort(key=lambda x: x['classified_date'], reverse=True)

        return render_template('folder_view.html',
                               folder_name=folder_name,
                               files=files_with_dates)

    except Exception as e:
        print(f"Erreur dans download_folder: {str(e)}")  # Debug
        return f"Erreur: {str(e)}", 500

# Route pour télécharger un fichier spécifique
@app.route('/download_file/<path:folder_name>/<path:filename>')
def download_specific_file(folder_name, filename):
    try:
        folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
        
        print(f"Téléchargement: {folder_path}/{filename}")  # Debug
        
        if not os.path.exists(folder_path):
            return "Dossier non trouvé", 404
            
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            return "Fichier non trouvé", 404
            
        return send_from_directory(folder_path, filename, as_attachment=True)
    except Exception as e:
        print(f"Erreur dans download_specific_file: {str(e)}")  # Debug
        return f"Erreur: {str(e)}", 500

# Route pour supprimer un fichier
@app.route('/delete_file/<path:folder_name>/<path:filename>', methods=['DELETE'])
def delete_file(folder_name, filename):
    try:
        folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
        file_path = os.path.join(folder_path, filename)

        print(f"Suppression demandée: {file_path}")  # Debug

        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Fichier non trouvé'}), 404

        # Supprimer le fichier
        os.remove(file_path)
        print(f"Fichier supprimé: {file_path}")

        # Supprimer les métadonnées associées
        metadata = load_metadata()
        cv_key = f"{folder_name}/{filename}"
        if cv_key in metadata:
            del metadata[cv_key]
            save_metadata(metadata)
            print(f"Métadonnées supprimées pour: {cv_key}")

        # Si le dossier est vide, le supprimer aussi
        if os.path.exists(folder_path) and len(os.listdir(folder_path)) == 0:
            os.rmdir(folder_path)
            print(f"Dossier vide supprimé: {folder_path}")

        return jsonify({'success': True, 'message': f'Fichier {filename} supprimé avec succès'})

    except Exception as e:
        print(f"Erreur dans delete_file: {str(e)}")  # Debug
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/classify', methods=['POST'])
def classify():
    results = []
    total = 0
    success = 0
    failed = 0

    if 'files' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400

    files = request.files.getlist('files')

    for file in files:
        total += 1
        if file.filename == '':
            results.append({'success': False, 'error': 'Nom de fichier vide'})
            failed += 1
            continue

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                text = classifier.extract_text_from_file(file_path)
                if len(text.strip()) == 0:
                    results.append({
                        'success': False,
                        'filename': filename,
                        'error': 'Impossible d\'extraire le texte du fichier'
                    })
                    failed += 1
                else:
                    profile, confidence, proba_dict = classifier.predict(text)

                    # Création du dossier directement dans 'downloads' (ex: downloads/Developpeur)
                    safe_profile = profile.replace('/', '_').strip()
                    profile_folder = os.path.join(CLASSIFIED_ROOT, safe_profile)
                    os.makedirs(profile_folder, exist_ok=True)

                    new_path = os.path.join(profile_folder, filename)
                    os.replace(file_path, new_path)
                    add_cv_metadata(safe_profile, filename) # Ajouter les métadonnées

                    results.append({
                        'success': True,
                        'filename': filename,
                        'profile': profile,
                        'confidence': confidence,
                        'probabilities': proba_dict,
                        'classified_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    success += 1

            except Exception as e:
                results.append({
                    'success': False,
                    'filename': filename,
                    'error': str(e)
                })
                failed += 1
        else:
            results.append({
                'success': False,
                'filename': file.filename,
                'error': 'Format non autorisé'
            })
            failed += 1

    return jsonify({
        'total': total,
        'success': success,
        'failed': failed,
        'results': results
    })

@app.route('/performance')
def performance():
    return "Page non trouvée", 404

if __name__ == '__main__':
    email_thread = Thread(target=start_email_monitoring, daemon=True)
    email_thread.start()
    print("Surveillance des emails démarrée en arrière-plan")
    app.run(debug=True, port=5001)