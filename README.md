# Reconnaissance faciale webcam (OpenCV)

Application Python qui utilise votre webcam pour **détecter les visages en direct** et afficher les infos des personnes reconnues : **nom, prénom, âge, sexe, info libre** — avec un score de confiance.

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV 4.8.1](https://img.shields.io/badge/OpenCV-4.8.1-green)

## Fonctionnalités

- **Détection en direct** des visages via la webcam (Haar Cascade).
- **Reconnaissance** des personnes enregistrées : affichage de `Prénom Nom | âge ans | sexe` + info libre + score de confiance, directement au-dessus du cadre.
- **Texte adaptatif** : la taille du texte et l'épaisseur du cadre sont proportionnelles à la taille du visage détecté.
- **Plein écran** (touche `F`) et fenêtre redimensionnable à la souris.
- **Enregistrement en direct** (touche `N`) : nom, prénom, âge, sexe, info libre.
- **Interface graphique** : glisser-déposer d'images directement dans la fenêtre, liste des personnes enregistrées avec leurs détails, et bouton **TERMINATE** (animation d'explosion sur le visage de la personne si elle est à l'écran).
- **Suivi stable** : le cadre et le texte ne « sautent » plus — position lissée, taille mise à jour au maximum toutes les 0,5 s avec une tolérance de ±12 % (les micro-variations sont ignorées).
- **Import intelligent** : un **sous-dossier = une personne** (toutes ses photos sont garanties lui appartenir, aucune confusion possible).
- **Multi-échantillons** : chaque nouvel échantillon améliore la reconnaissance (ré-entraînement automatique à chaque ajout).
- **Redimensionnement sans déformation** : la vidéo s'adapte à la taille de la fenêtre en conservant ses proportions (bandes noires type « letterbox »).
- **Source vidéo au choix** : webcam 0/1/2… ou fichier vidéo (mp4, avi, mov…) via l'interface.
- **Genre libre** : le champ genre est un champ libre (H, F, NB, transfem…) que vous définissez vous-même à l'enregistrement — la base de données supporte toutes les catégories.
- **Ressemblance avec des célébrités** : déposez des photos dans `celebrites/` (un sous-dossier par célébrité) et le programme affiche la célébrité ressemblante avec un score.
- **Texte surligné** : chaque texte affiché a un fond noir (comme surligné) → toujours lisible, même sur fond clair.
- **DeepFace (optionnel)** : si installé (`pip install deepface`), estime **âge apparent, genre apparent et émotion** de chaque visage, affichés sous le cadre en bleu clair.

---

## Installation pas à pas

### Étape 0 — Prérequis

- **Python 3.8 ou plus récent** : [python.org/downloads](https://www.python.org/downloads/)
  - **Windows** : lors de l'installation, cochez bien **« Add Python to PATH »** (écran d'accueil de l'installeur).
- Une **webcam** fonctionnelle.
- `git` (facultatif si vous téléchargez le ZIP du dépôt) : [git-scm.com](https://git-scm.com/)

### Étape 1 — Récupérer le projet

```bash
git clone https://github.com/Massicotte/face_reco.git
cd face_reco
```

> Sans git : sur la page du dépôt → bouton vert **Code → Download ZIP** → extrayez l'archive puis ouvrez un terminal dans le dossier extrait.

### Étape 2 — (Recommandé) Créer un environnement virtuel

Cela évite les conflits de versions avec d'autres projets Python.

**Windows :**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Sous Windows PowerShell, si l'activation est bloquée :
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` puis réessayez.

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

Cela installe :
- `opencv-contrib-python==4.8.1.78` — **important** : la version *contrib* est requise, car le module de reconnaissance `cv2.face` n'existe pas dans le paquet `opencv-python` standard. La version est figée (4.8.1.78) car c'est une version éprouvée et stable.
- `numpy>=1.21,<2` — numpy 2.x peut causer des incompatibilités avec cette version d'OpenCV.

> **Si vous aviez déjà `opencv-python` installé** (conflit classique) :
> ```bash
> pip uninstall opencv-python opencv-contrib-python -y
> pip install -r requirements.txt
> ```

### Étape 4 — Lancer

```bash
python face_reco.py
```

Une fenêtre s'ouvre avec le flux webcam. Dans le terminal s'affiche :
```
Webcam activee. Touches : N = nouvel enregistrement | F = plein ecran | Q = quitter
Deposez des images dans ./a_importer/ pour les importer (traitement automatique).
```

✅ **C'est installé !** Passez à « Utilisation ».

---

## Utilisation

### Commandes clavier

| Touche | Action |
|---|---|
| `N` | Enregistrer la personne montrée (un seul visage à la caméra) |
| `F` | Basculer plein écran ↔ fenêtre |
| `Q` | Quitter |

### Enregistrer quelqu'un (webcam)

1. La personne se place seule face à la caméra.
2. Appuyez sur **`N`**.
3. Répondez aux questions dans le **terminal** : Nom, Prénom, Âge, Sexe (H/F), Info complémentaire (facultatif).
4. Le visage est capturé et la personne apparaît désormais en **vert** avec ses infos.

Pour améliorer la reconnaissance d'une personne : appuyez plusieurs fois sur **`N`** sous des angles et éclairages différents — si le visage est déjà reconnu, seul un nouvel échantillon est ajouté, sans redemander les infos.

### L'interface graphique (fenêtre « Face Reco - Contrôle »)

Au lancement, **deux fenêtres** s'ouvrent : la vidéo webcam et le panneau de contrôle.

- **Zone de dépôt** : glissez-déposez directement vos images ou dossiers dessus (nécessite `tkinterdnd2`, installé par `requirements.txt`).
- **Source vidéo** : boutons « Fichier video... » (analyse un fichier mp4/avi/mov...), « Changer de webcam » (numéro 0, 1, 2...), « Retour webcam 0 ». La vidéo s'affiche toujours sans déformation, quelle que soit la taille de la fenêtre.
- **Liste des personnes** : toutes les personnes enregistrées. Cliquez sur un nom → ses détails s'affichent (nom, prénom, âge, genre, info, date d'enregistrement).
- **Bouton TERMINATE** : apparaît (activé) quand une personne est sélectionnée. Au clic, le programme attendjusqu'à 5 s que cette personne soit reconnue à l'écran → **animation d'explosion** sur son visage 🎆
- **Messages de statut** : résultat des imports, confirmations, erreurs — plus besoin de regarder le terminal.

### Comment le programme sait à qui appartient chaque image ?

**Un sous-dossier = une personne.** C'est la règle d'or :

```
a_importer/
├── Jean_Dupont/          ← TOUTES ces photos = Jean Dupont (garanti)
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Marie_Martin/          ← TOUTES ces photos = Marie Martin
│   └── selfie.jpg
└── photo_orpheline.jpg    ← image isolée : rattachée par reconnaissance, ou nouvelle personne
```

- Si le nom du sous-dossier correspond à une **personne déjà enregistrée** → les échantillons s'ajoutent à elle.
- Sinon → le programme demande ses infos via une **boîte de dialogue** dans la fenêtre de contrôle.
- Les images isolées (posées à la racine) sont rattachées **par reconnaissance** : personne déjà connue → nouveléchantillon ; visage inconnu → demande d'infos.

> ✅ Avec les sous-dossiers, **aucune confusion possible** : toutes les photos d'un dossier vont à la même personne, sous tous les angles voulus.
> Les images traitées sont déplacées dans `a_importer/traite/`.

### Genre : un champ libre (pas une devinette)

Le champ **genre est libre** : H, F, NB, transfem, non-binaire, ce que vous voulez — vous le définissez à
l'enregistrement et il s'affiche avec le visage. C'est un choix volontaire : un classifieur ne peut pas
« deviner » une identité de genre à partir d'un visage de façon fiable ; la valeur que vous saisissez
est donc la source de vérité, stockée dans `data/people.json`.

### Ressemblance avec des célébrités

Créez un dossier `celebrites/` à côté du script, avec **un sous-dossier par célébrité** :

```
celebrites/
├── Jean_Dujardin/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Scarlett_Johansson/
    └── photo.jpg
```

- Le programme construit un modèle de comparaison à partir de ces photos.
- Sous chaque visage détecté (connu ou inconnu), il affiche la célébrité la plus ressemblante :
  `~ Jean Dujardin (62%)` en violet.
- Astuce : glissez directement un dossier nommé `celebrites-...` dans la zone de dépôt, il sera
  reconnu comme célébrité et le modèle recalculé.
- Plus vous mettez de photos par célébrité, meilleure est la comparaison. C'est un jeu de
  ressemblance approximatif — pas une science exacte.

### Ce qui s'affiche

- **Cadre vert** + `Prénom Nom | âge ans | sexe` : personne reconnue. En dessous : info libre + score de confiance (`confiance: NN`, plus c'est haut, mieux c'est).
- **Cadre rouge** + `Inconnu` : visage non reconnu (ou score trop faible).

---

## Données

| Fichier / dossier | Contenu |
|---|---|
| `data/people.json` | Infos des personnes (nom, prénom, âge, sexe, info, date d'enregistrement) |
| `data/faces/` | Photos des visages utilisées pour l'entraînement (`label_0.png`, `label_1.png`, …) |
| `a_importer/` | Vos images à importer (glisser-déposer) |
| `a_importer/traite/` | Images déjà traitées |

> ⚠️ **Vie privée** : `data/` contient des photos de visages et des informations personnelles. Ce dossier est exclu de git via `.gitignore` — ne le partagez pas.

Pour **supprimer** une personne : retirez ses lignes de `data/people.json` et ses fichiers `label_N.png` dans `data/faces/`, puis relancez le programme.

Pour **réinitialiser** tout : supprimez le dossier `data/`.

---

## Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | `pip install -r requirements.txt` (et vérifiez que l'environnement virtuel est actif) |
| `AttributeError: module 'cv2' has no attribute 'face'` | Vous avez `opencv-python` au lieu de la version *contrib* : `pip uninstall opencv-python opencv-contrib-python -y` puis `pip install -r requirements.txt` |
| `cv2.error: ... DLL` (Windows) | Conflit de versions : désinstallez tous les paquets opencv (voir ci-dessus) et réinstallez |
| « Erreur : impossible d'accéder à la webcam » | Webcam utilisée par une autre appli (Zoom, Teams…) — fermez-la. Windows : Paramètres → Confidentialité → Caméra → autorisez Python/le terminal |
| `numpy` incompatible | `pip install "numpy>=1.21,<2"` |
| La détection ne trouve pas les visages | Éclairage insuffisant, visage trop loin (< 80px) ou pas de face. Améliorez la lumière |
| Reconnaissance confond deux personnes | Ajoutez plus d'échantillons pour chacune (touche `N` ou import d'images) |

---

## Désinstallation

Tout est contenu dans le dossier du projet — rien n'est installé ailleurs
(pas de registre, pas de fichiers système). Trois étapes :

1. **Quitter le programme** si elle tourne (touche `Q`), et fermer le terminal.
2. **Supprimer l'environnement virtuel** (facultatif, il sera de toute façon
   supprimé avec le dossier) :
   ```bash
   deactivate                    # si l'environnement est encore actif
   rm -rf .venv                  # macOS / Linux
   rmdir /s /q .venv             # Windows (cmd)
   Remove-Item -Recurse -Force .venv   # Windows (PowerShell)
   ```
3. **Supprimer le dossier du projet** (depuis le dossier parent) :
   ```bash
   rm -rf face_reco                          # macOS / Linux
   rmdir /s /q face_reco                     # Windows — invite de commandes (cmd)
   Remove-Item -Recurse -Force face_reco      # Windows — PowerShell
   ```

   > Note : `rmdir /s /q` ne fonctionne **pas** dans PowerShell (erreur
   > « paramètre positionnel ») — utilisez `Remove-Item` dans PowerShell.

> ⚠️ Cette étape supprime aussi `data/` : **toutes les photos de visages et
> les infos des personnes enregistrées sont définitivement effacées.**
>
> Pour tout supprimer **sauf les données** : déplacez le dossier `data/`
> ailleurs avant de supprimer le dossier du projet.

Si vous n'avez **pas** utilisé d'environnement virtuel, nettoyez aussi les
paquets installés globalement :

```bash
pip uninstall opencv-contrib-python numpy -y
```

Et pour ne plus le voir sur GitHub : *Settings → Danger Zone → Delete this
repository* sur la page du dépôt.

---

## Limites

- Détection par Haar Cascade : sensible à l'éclairage, visage de face de préférence.
- L'**âge et le genre sont saisis manuellement** à l'enregistrement — le programme ne les devine pas (et ne peut pas le faire de façon fiable).
- La ressemblance avec les célébrités est **indicative** (algorithme LBPH, pas de deep learning).
- **DeepFace** est optionnel : sans lui, tout le reste fonctionne. Avec lui, les estimations d'âge/genre/émotion sont des **estimations automatiques** — l'âge et le genre que VOUS saisissez à l'enregistrement restent la source de vérité affichée en priorité. L'installation de DeepFace entraîne TensorFlow (~1 Go).
- DeepFace analyse chaque visage toutes les 5 secondes (pas à chaque image) pour préserver la fluidité.
- Reconnaissance par LBPH : fonctionne bien avec quelques échantillons, mais reste un algorithme classique (pas de deep learning).

## Licence

Projet personnel — libre d'utilisation et de modification.
