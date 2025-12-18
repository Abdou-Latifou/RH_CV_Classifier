import os
import imapclient
import email
from email.header import decode_header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import tempfile
import schedule
import json
import time
from datetime import datetime
from classifier import CVClassifier
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv()

# Configuration email
EMAIL_USER = os.getenv('EMAIL_USER', 'recrutement@wearedigijob.com')  # Votre email
EMAIL_PASS = os.getenv('EMAIL_PASS', '')  # Mot de passe d'app Gmail

# Fichier pour tracker les emails traités
PROCESSED_EMAILS_FILE = 'processed_emails.txt'




def add_cv_metadata(profile, filename):
    """Ajoute les métadonnées du CV (date/heure) pour l'affichage dans l'app"""
    metadata_file = 'cv_metadata.json'
    metadata = {}
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    cv_key = f"{profile}/{filename}"
    metadata[cv_key] = {
        'profile': profile,
        'filename': filename,
        'classified_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'active'
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)





def load_processed_emails():
    """Charge la liste des emails déjà traités"""
    if os.path.exists(PROCESSED_EMAILS_FILE):
        with open(PROCESSED_EMAILS_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_email(msg_id):
    """Sauvegarde un email comme traité"""
    with open(PROCESSED_EMAILS_FILE, 'a') as f:
        f.write(f"{msg_id}\n")

# Détecter le fournisseur email et configurer les serveurs
email_domain = EMAIL_USER.split('@')[1].lower()

if email_domain == 'gmail.com':
    IMAP_SERVER = 'imap.gmail.com'
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
elif email_domain == 'wearedigijob.com':
    # Configuration spécifique pour wearedigijob.com (hébergé chez O2switch)
    IMAP_SERVER = 'mail.wearedigijob.com'
    SMTP_SERVER = 'mail.wearedigijob.com'
    SMTP_PORT = 465  # SSL
else:
    # Pour O2switch et autres hébergeurs français
    IMAP_SERVER = 'mail.o2switch.net'
    SMTP_SERVER = 'smtp.o2switch.net'
    SMTP_PORT = 587  # TLS
    logging.info(f"Utilisation des serveurs O2switch génériques pour le domaine {email_domain}")

logging.info(f"Configuration email pour {email_domain}: IMAP={IMAP_SERVER}, SMTP={SMTP_SERVER}:{SMTP_PORT}")

# Initialiser le classifieur
classifier = CVClassifier()

def load_classifier():
    """Charge le modèle entraîné"""
    if os.path.exists('cv_classifier_model.pkl'):
        classifier.load_model()
        logging.info("Modèle chargé pour le traitement email")
        return True
    else:
        logging.error("Modèle non trouvé. Veuillez entraîner le modèle d'abord.")
        return False

def extract_text_from_attachment(part, filename):
    """Extrait le texte d'une pièce jointe en utilisant les méthodes du classifieur"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(part.get_payload(decode=True))
        temp_file_path = temp_file.name

    try:
        text = classifier.extract_text_from_file(temp_file_path)
        return text
    finally:
        os.unlink(temp_file_path)

def process_attachment(part, filename, sender_email):
    """Traite une pièce jointe CV"""
    logging.info(f"Traitement de la pièce jointe: {filename} de {sender_email}")

    # Extraire le texte
    cv_text = extract_text_from_attachment(part, filename)

    if not cv_text or len(cv_text.strip()) < 50:
        logging.warning(f"Texte extrait trop court ou vide pour {filename}")
        return None

    # Classifier le CV
    try:
        prediction, confidence, probabilities = classifier.predict(cv_text)
        logging.info(f"Classification réussie: {prediction} (confiance: {confidence:.2f})")

        # Sauvegarder le fichier dans le dossier classifié
        safe_profile = prediction.replace('/', '_').strip()
        classified_dir = os.path.join('downloads', safe_profile)
        os.makedirs(classified_dir, exist_ok=True)
        
        file_path = os.path.join(classified_dir, filename)
        if os.path.exists(file_path):
            logging.info(f"CV déjà traité et classé: {file_path}")
            return None  # Ne pas retraiter
        
        with open(file_path, 'wb') as f:
            f.write(part.get_payload(decode=True))
        logging.info(f"Fichier sauvegardé: {file_path}")

        result = {
            'filename': filename,
            'sender': sender_email,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities,
            'processed_at': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        logging.error(f"Erreur lors de la classification de {filename}: {e}")
        return None

def send_response_email(recipient, subject, body):
    """Envoie un email de réponse (optionnel)"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        if SMTP_PORT == 465:
            # Port SSL
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
            server.login(EMAIL_USER, EMAIL_PASS)
        else:
            # Port TLS (587)
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
        
        text = msg.as_string()
        server.sendmail(EMAIL_USER, recipient, text)
        server.quit()

        logging.info(f"Email de réponse envoyé à {recipient}")

    except Exception as e:
        logging.error(f"Erreur envoi email à {recipient}: {e}")

def check_emails():
    """Vérifie les nouveaux emails et traite les pièces jointes CV"""
    try:
        server = imapclient.IMAPClient(IMAP_SERVER, ssl=True)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.select_folder('INBOX')

        # Rechercher les emails d'aujourd'hui
        from datetime import date
        today = date.today().strftime('%d-%b-%Y')
        messages = server.search(['SINCE', today])

        if not messages:
            logging.info("Aucun nouvel email trouvé")
            return

        logging.info(f"{len(messages)} emails trouvés aujourd'hui")

        for msg_id in messages:
            try:
                # Petite pause pour éviter de surcharger le serveur
                time.sleep(2)
                # Récupérer l'email
                raw_msg = server.fetch([msg_id], ['BODY[]', 'ENVELOPE'])[msg_id]
                envelope = raw_msg[b'ENVELOPE']
                sender = envelope.from_[0].mailbox.decode() + '@' + envelope.from_[0].host.decode()

                msg = email.message_from_bytes(raw_msg[b'BODY[]'])

                # Traiter les pièces jointes
                attachments_processed = 0

                for part in msg.walk():
                    if part.get_content_disposition() == 'attachment':
                        filename = part.get_filename()
                        if filename:
                            # Décoder le nom du fichier si nécessaire
                            decoded_filename = decode_header(filename)[0][0]
                            if isinstance(decoded_filename, bytes):
                                decoded_filename = decoded_filename.decode()

                            # Vérifier l'extension
                            allowed_extensions = {'.pdf', '.txt', '.docx', '.pptx', '.xlsx', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.doc', '.rtf'}
                            if any(decoded_filename.lower().endswith(ext) for ext in allowed_extensions):
                                result = process_attachment(part, decoded_filename, sender)

                                if result:
                                    attachments_processed += 1

                                    # Loguer le résultat
                                    logging.info(f"CV classé: {result['filename']} -> {result['prediction']} ({result['confidence']:.1f}%)")

                                    # Optionnel: envoyer un email de réponse
                                    # subject = f"Classification de votre CV: {result['prediction']}"
                                    # body = f"Votre CV '{result['filename']}' a été classé comme '{result['prediction']}' avec une confiance de {result['confidence']:.1f}%."
                                    # send_response_email(sender, subject, body)

                if attachments_processed > 0:
                    logging.info(f"{attachments_processed} CV(s) traité(s) dans l'email de {sender}")

            except Exception as e:
                logging.error(f"Erreur traitement email {msg_id}: {e}")

        server.logout()

    except Exception as e:
        logging.error(f"Erreur connexion IMAP: {e}")

def start_email_monitoring():
    """Démarre la surveillance des emails"""
    if not load_classifier():
        return

    logging.info("Démarrage de la surveillance des emails...")

    # Vérifier immédiatement au démarrage
    check_emails()

    # Planifier les vérifications toutes les 5 minutes
    schedule.every(5).minutes.do(check_emails)

    while True:
        schedule.run_pending()
        time.sleep(60)  # Vérifier toutes les minutes si une tâche est due

if __name__ == '__main__':
    start_email_monitoring()