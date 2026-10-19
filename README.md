# Reconnaissance faciale webcam (OpenCV)

Application Python qui utilise votre webcam pour **détecter les visages en direct** et afficher les infos des personnes reconnues : **nom, prénom, âge, sexe, info libre** — avec un score de confiance.

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV 4.8.1](https://img.shields.io/badge/OpenCV-4.8.1-green)

## Fonctionnalités

- **Détection en direct** des visages via la webcam (Haar Cascade).
- **Reconnaissance** des personnes enregistrées : affichage de `Prénom Nom | âge ans | sexe` + info libre + score de confiance, directement au-dessus du cadre.
- **Texte adaptatif** : la taille du texte et l'épaisseur du cadre sont proportionnelles à la taille du visage détecté.
- **Plein écran** (touche `F`) et fenêtre redimensionnable à la souris.
- **Enregistrement en direct** (touche `N`) : nom, prénom, âge, sexe, info libre.
- **Import d'images en glisser-déposer** (dossier `a_importer/`) : ajoutez plusieurs photos de la même personne sous différents angles, le programme les rattache automatiquement.
- **Multi-échantillons** : chaque nouvel échantillon améliore la reconnaissance (ré-entraînement automatique à chaque ajout).

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

### Importer des photos (glisser-déposer)

Les fenêtres OpenCV n'acceptent pas le glisser-déposer natif du système : le programme surveille donc en continu le dossier **`a_importer/`** (créé automatiquement à côté du script), toutes les ~2 secondes, même pendant que la webcam tourne.

1. Glissez vos photos dans `a_importer/` (png, jpg, jpeg, bmp, webp).
2. Chaque visage détecté est analysé :
   - **Personne déjà connue** → ajouté directement comme nouvel échantillon (nouvel angle).
   - **Personne inconnue** → le programme demande ses infos dans le terminal, puis les images suivantes de la même personne sont **rattachées automatiquement**, sans rien redemander.
3. Les images traitées sont déplacées dans `a_importer/traite/`.

> **Astuce** : mélangez photos de face, de trois-quarts, avec/sans lunettes, éclairages variés — c'est ce qui fait la différence pour la reconnaissance sous plusieurs angles.

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
   deactivate          # si l'environnement est encore actif
   rm -rf .venv        # macOS / Linux
   rmdir /s /q .venv   # Windows (cmd)
   ```
3. **Supprimer le dossier du projet** :
   ```bash
   cd ..
   rm -rf face_reco        # macOS / Linux
   rmdir /s /q face_reco   # Windows (cmd)
   ```

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
- L'**âge et le sexe sont saisis manuellement** à l'enregistrement — le programme ne les devine pas.
- Reconnaissance par LBPH : fonctionne bien avec quelques échantillons, mais reste un algorithme classique (pas de deep learning).

## Licence

Projet personnel — libre d'utilisation et de modification.
