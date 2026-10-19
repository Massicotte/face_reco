# Reconnaissance faciale webcam (OpenCV)

Application Python qui utilise votre webcam pour détecter les visages et afficher
les infos des personnes enregistrées : **nom, prénom, âge, sexe, info libre**.

## Installation (sur votre PC)

```bash
pip install -r requirements.txt
```

> Note : il faut le paquet `opencv-contrib-python` (et non `opencv-python`), car le
> module de reconnaissance faciale `cv2.face` n'est que dans la version "contrib".

## Utilisation

```bash
python face_reco.py
```

1. Une fenêtre s'ouvre avec le flux de la webcam, les visages sont encadrés en vert.
2. **Touche `N`** : enregistre une nouvelle personne → le programme vous demande
   nom, prénom, âge, sexe et une info libre, puis prend une photo du visage.
   (Montrez **un seul** visage à la caméra au moment d'appuyer.)
3. Ensuite, chaque visage reconnu affiche ses infos au-dessus du cadre :
   `Prénom Nom | âge | sexe`, plus l'info libre et un score de confiance.
4. Visage inconnu → affiché en rouge avec « Inconnu ».
5. **Touche `Q`** : quitter.

## Données

- `data/people.json` : les infos des personnes.
- `data/faces/` : les photos de visages utilisées pour l'entraînement.

Améliorer la reconnaissance : appuyez plusieurs fois sur `N` pour la même
personne (sous des angles/éclairages différents) — chaque échantillon est
enregistré et le modèle se ré-entraîne à chaque fois. Si le visage reconnu
correspond à une personne existante, seul un nouvel échantillon est ajouté,
sans redemander les infos.

## Versions recommandées

Si les versions récentes d'OpenCV posent problème (`cv2.face` introuvable,
DLL manquante, etc.), utilisez ces versions éprouvées :

```
opencv-contrib-python==4.8.1.78
numpy>=1.21,<2
```

## Limites

- Détection avec Haar Cascade : sensible à l'éclairage, visage de face requis.
- L'âge et le sexe sont **saisis manuellement**, pas devinés par le programme.
