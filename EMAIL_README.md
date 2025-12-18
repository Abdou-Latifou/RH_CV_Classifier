# Extension Email pour CV Classifier

Cette extension permet à votre application de traiter automatiquement les CV reçus par email.

## Configuration

### Pour Gmail :
1. **Activer IMAP sur Gmail :**
   - Allez dans les paramètres Gmail > Voir tous les paramètres > Onglet "Transfert et POP/IMAP"
   - Activez "Activer IMAP"

2. **Générer un mot de passe d'application :**
   - Allez dans https://myaccount.google.com/security
   - Activer la vérification en 2 étapes si ce n'est pas fait
   - Générer un mot de passe d'app pour "Mail"
   - Copiez ce mot de passe (16 caractères)

3. **Configurer le fichier .env :**
   - Ouvrez le fichier `.env`
   - Mettez EMAIL_USER=votre@gmail.com
   - Remplacez EMAIL_PASS= par le mot de passe d'app généré

### Pour O2switch :
1. **Votre email doit être hébergé chez O2switch** (ex: contact@votre-domaine.fr)
2. **IMAP est déjà configuré** chez O2switch
3. **Configurer le fichier .env :**
   - Mettez EMAIL_USER=votre@domaine.fr
   - Mettez EMAIL_PASS=votre_mot_de_passe_o2switch

### Pour autres fournisseurs :
L'app détecte automatiquement les serveurs IMAP/SMTP basés sur votre domaine email.

## Fonctionnement

- L'app vérifie automatiquement les nouveaux emails toutes les 5 minutes
- Elle traite les pièces jointes de type PDF, DOCX, TXT, etc.
- Les CV sont classés automatiquement avec le modèle ML
- Les résultats sont loggés dans la console

## Utilisation

Envoyez un email avec un CV en pièce jointe à `kedjlate.sn@gmail.com`. L'app le traitera automatiquement.

## Sécurité

- Seuls les emails avec pièces jointes autorisées sont traités
- Les fichiers temporaires sont supprimés après traitement
- Utilisez un mot de passe d'app, pas votre mot de passe principal

## Test

Pour tester, envoyez un email avec un CV à votre adresse, puis vérifiez les logs de l'app.