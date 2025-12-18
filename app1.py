import re
import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
from flask import Flask, render_template_string, request, jsonify, send_from_directory
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



HTML_TEMPLATE = """ 
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification de CV - IA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: url('/static/logo.png') no-repeat center center fixed; background-size: cover; min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .card { background: white; border-radius: 20px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); margin-bottom: 30px; }
        .upload-zone { border: 3px dashed #667eea; border-radius: 15px; padding: 60px 20px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8f9ff; }
        .upload-zone:hover { border-color: #764ba2; background: #f0f1ff; transform: scale(1.02); }
        .upload-zone.dragover { background: #e8ebff; border-color: #764ba2; }
        .upload-icon { font-size: 4em; margin-bottom: 20px; }
        .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 15px 40px; border-radius: 50px; font-size: 1.1em; cursor: pointer; transition: transform 0.2s; margin: 10px; }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .results { margin-top: 30px; }
        .result-card { background: #f8f9ff; border-radius: 15px; padding: 25px; margin-bottom: 20px; border-left: 5px solid #667eea; }
        .profile-badge { display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 25px; border-radius: 50px; font-weight: bold; font-size: 1.2em; margin: 10px 0; }
        .confidence-bar { background: #e0e0e0; border-radius: 10px; height: 30px; margin: 15px 0; overflow: hidden; position: relative; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 1s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }
        .probabilities { margin-top: 20px; }
        .prob-item { display: flex; align-items: center; margin: 10px 0; padding: 10px; background: white; border-radius: 8px; }
        .prob-label { flex: 0 0 300px; font-weight: 500; }
        .prob-bar { flex: 1; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; margin: 0 15px; }
        .prob-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); transition: width 0.8s ease; }
        .prob-value { flex: 0 0 60px; text-align: right; font-weight: bold; color: #667eea; }
        .loading { display: none; text-align: center; padding: 40px; }
        .spinner { border: 5px solid #f3f3f3; border-top: 5px solid #667eea; border-radius: 50%; width: 60px; height: 60px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; }
        .stat-value { font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }
        .stat-label { opacity: 0.9; font-size: 0.9em; }
        .file-list { margin-top: 20px; }
        .file-item { display: flex; align-items: center; padding: 15px; background: #f8f9ff; border-radius: 10px; margin: 10px 0; }
        .file-icon { font-size: 2em; margin-right: 15px; }
        .file-info { flex: 1; }
        .file-name { font-weight: bold; color: #333; }
        .file-size { color: #666; font-size: 0.9em; }
        .download-section { background: #e8f5e9; padding: 25px; border-radius: 15px; margin: 20px 0; border-left: 5px solid #4caf50; }
        .folder-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
        .folder-card { background: white; padding: 20px; border-radius: 10px; text-align: center; cursor: pointer; transition: transform 0.2s; border: 2px solid #4caf50; }
        .folder-card:hover { transform: scale(1.05); background: #f1f8f4; }
        .folder-icon { font-size: 3em; margin-bottom: 10px; }
        .folder-name { font-weight: bold; color: #333; margin-bottom: 5px; }
        .folder-count { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Classification de CV par IA</h1>
            <p>Uploadez vos CV et laissez l'IA les classifier automatiquement</p>
            <p>selon les profils professionnels.</p>
        </div>

        <div class="card">
            <div class="download-section">
                <h2>📥 Espace Téléchargement RH</h2>
                <p style="margin: 15px 0;">Cliquez sur un dossier pour télécharger les CV classifiés par profil :</p>
                <div class="folder-grid" id="folderGrid">
                    <div style="text-align: center; padding: 40px; grid-column: 1/-1; color: #666;">
                        Chargement des dossiers...
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>📤 Uploader vos CV</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <div class="upload-zone" id="dropZone">
                    <div class="upload-icon">📄</div>
                    <h3>Glissez-déposez vos CV ici</h3>
                    <p>ou cliquez pour sélectionner des fichiers</p>
                    <p style="color: #666; margin-top: 10px;">Formats acceptés: PDF, TXT, DOCX, PPTX, XLSX, JPG, PNG</p>
                    <input type="file" id="fileInput" name="files" multiple accept=".pdf,.txt,.docx,.pptx,.xlsx,.jpg,.jpeg,.png,.gif,.bmp,.tiff" style="display: none;">
                </div>
                <div class="file-list" id="fileList"></div>
                <div style="text-align: center; margin-top: 20px;">
                    <button type="submit" class="btn" id="classifyBtn" disabled> 🚀 Classifier les CV </button>
                </div>
            </form>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Classification en cours...</p>
            </div>
        </div>
        <div id="results" class="results"></div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileList = document.getElementById('fileList');
        const uploadForm = document.getElementById('uploadForm');
        const classifyBtn = document.getElementById('classifyBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const folderGrid = document.getElementById('folderGrid');
        let selectedFiles = [];

        // Charger les dossiers disponibles au chargement de la page
        loadFolders();

        async function loadFolders() {
            try {
                const response = await fetch('/get_folders');
                const data = await response.json();
                displayFolders(data.folders);
            } catch (error) {
                folderGrid.innerHTML = '<div style="text-align: center; padding: 40px; grid-column: 1/-1; color: #f00;">Erreur de chargement des dossiers</div>';
            }
        }

        function displayFolders(folders) {
            if (folders.length === 0) {
                folderGrid.innerHTML = '<div style="text-align: center; padding: 40px; grid-column: 1/-1; color: #666;">Aucun CV classifié pour le moment</div>';
                return;
            }

            folderGrid.innerHTML = '';
            folders.forEach(folder => {
                const folderCard = document.createElement('div');
                folderCard.className = 'folder-card';
                folderCard.innerHTML = `
                    <div class="folder-icon">📁</div>
                    <div class="folder-name">${folder.name}</div>
                    <div class="folder-count">${folder.count} CV</div>
                `;
                folderCard.onclick = () => downloadFolder(folder.name);
                folderGrid.appendChild(folderCard);
            });
        }

        async function downloadFolder(folderName) {
            window.open(`/download_folder/${encodeURIComponent(folderName)}`, '_blank');
        }

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
        dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
        fileInput.addEventListener('change', (e) => { handleFiles(e.target.files); });

        function handleFiles(files) {
            selectedFiles = Array.from(files);
            displayFiles();
            classifyBtn.disabled = selectedFiles.length === 0;
        }

        function displayFiles() {
            fileList.innerHTML = '';
            selectedFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div class="file-icon">📄</div>
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-size">${(file.size / 1024).toFixed(2)} KB</div>
                    </div>
                `;
                fileList.appendChild(fileItem);
            });
        }

        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData();
            selectedFiles.forEach(file => { formData.append('files', file); });
            loading.style.display = 'block';
            results.innerHTML = '';
            classifyBtn.disabled = true;

            try {
                const response = await fetch('/classify', { method: 'POST', body: formData });
                const data = await response.json();
                displayResults(data);
                loadFolders(); // Recharger les dossiers après classification
            } catch (error) {
                results.innerHTML = `
                    <div class="card" style="background: #fee; border-left: 5px solid #f00;">
                        <h3>❌ Erreur</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            } finally {
                loading.style.display = 'none';
                classifyBtn.disabled = false;
            }
        });

        function displayResults(data) {
            let html = '<div class="card"><h2>✅ Résultats de Classification</h2>';
            html += '<div class="stats-grid">';
            html += `
                <div class="stat-card">
                    <div class="stat-value">${data.total}</div>
                    <div class="stat-label">CV Classifiés</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.success}</div>
                    <div class="stat-label">Réussis</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.failed}</div>
                    <div class="stat-label">Échoués</div>
                </div>
            `;
            html += '</div>';

            data.results.forEach(result => {
                if (result.success) {
                    html += `
                        <div class="result-card">
                            <h3>📄 ${result.filename}</h3>
                            <div class="profile-badge">${result.profile}</div>
                            <h4 style="margin-top: 20px;">Confiance:</h4>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${result.confidence * 100}%">
                                    ${(result.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                            <h4>Probabilités par profil:</h4>
                            <div class="probabilities">
                    `;

                    const sortedProbs = Object.entries(result.probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 5);

                    sortedProbs.forEach(([prof, prob]) => {
                        html += `
                            <div class="prob-item">
                                <div class="prob-label">${prof}</div>
                                <div class="prob-bar">
                                    <div class="prob-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
                            </div>
                        `;
                    });

                    html += ` </div></div> `;
                } else {
                    html += `
                        <div class="result-card" style="border-left-color: #f00;">
                            <h3>❌ ${result.filename}</h3>
                            <p style="color: #f00;">${result.error}</p>
                        </div>
                    `;
                }
            });

            html += '</div>';
            results.innerHTML = html;
        }
    </script>
</body>
</html>
"""

PERFORMANCE_TEMPLATE = """ 
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance du Modèle</title>
    <style>
        /* ... (style inchangé) ... */
    </style>
</head>
<body>
    <!-- ... (contenu inchangé) ... -->
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Route pour obtenir la liste des dossiers et leur contenu
@app.route('/get_folders')
def get_folders():
    folders = []
    try:
        if os.path.exists(CLASSIFIED_ROOT):
            for folder_name in os.listdir(CLASSIFIED_ROOT):
                folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
                if os.path.isdir(folder_path):
                    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                    folders.append({
                        'name': folder_name,
                        'count': len(files)
                    })
    except Exception as e:
        print(f"Erreur lors de la récupération des dossiers: {e}")
    
    return jsonify({'folders': folders})

# Route pour télécharger un fichier spécifique d'un dossier
@app.route('/download_folder/<path:folder_name>')
def download_folder(folder_name):
    try:
        # Ne pas utiliser secure_filename ici car il supprime les underscores
        folder_path = os.path.join(CLASSIFIED_ROOT, folder_name)
        
        print(f"Tentative d'accès au dossier: {folder_path}")  # Debug
        
        if not os.path.exists(folder_path):
            return f"Dossier non trouvé: {folder_path}", 404
            
        if not os.path.isdir(folder_path):
            return "Ce n'est pas un dossier valide", 400
        
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        if not files:
            return "Aucun fichier dans ce dossier", 404
        
        # Si un seul fichier, le télécharger directement
        #if len(files) == 1:
        #    return send_from_directory(folder_path, files[0], as_attachment=True)
        
        # Sinon, créer une page HTML listant tous les fichiers
        html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <title>Téléchargement - {folder_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
                .file-list {{ list-style: none; padding: 0; }}
                .file-item {{ padding: 15px; margin: 10px 0; background: #f8f9ff; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; }}
                .file-name {{ font-weight: bold; color: #333; }}
                .btn {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 10px 20px; border-radius: 25px; cursor: pointer; text-decoration: none; display: inline-block; }}
                .btn:hover {{ opacity: 0.9; }}
                .back-btn {{ background: #666; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="btn back-btn">← Retour</a>
                <h1>📁 {folder_name}</h1>
                <p>Cliquez sur un fichier pour le télécharger :</p>
                <ul class="file-list">
        """
        
        for file in files:
            # Encoder le nom du dossier et du fichier pour l'URL
            encoded_folder = folder_name.replace('/', '%2F')
            encoded_file = file.replace(' ', '%20')
            html += f"""
                <li class="file-item">
                    <span class="file-name">📄 {file}</span>
                    <a href="/download_file/{encoded_folder}/{encoded_file}" class="btn">⬇️ Télécharger</a>
                </li>
            """
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        return html
        
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