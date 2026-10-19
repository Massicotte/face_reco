# Dossier celebrites/

Déposez ici les photos de célébrités : **un sous-dossier par célébrité**.

```
celebrites/
├── Jean_Dujardin/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Scarlett_Johansson/
│   └── photo.jpg
└── Nom_De_La_Celebrite_1/   <- modèle à renommer (ou supprimer)
```

## Comment ça marche

1. Renommez `Nom_De_La_Celebrite_1` avec le nom de votre célébrité (les `_`
   seront affichés comme des espaces).
2. Glissez-y ses photos (3-5 photos sous des angles différents = meilleur résultat).
3. Au prochain lancement du programme (ou après un glisser-déposer dans la
   zone de dépôt de l'interface), le modèle de comparaison est reconstruit.
4. Sous chaque visage détecté, le programme affiche la célébrité la plus
   ressemblante : `~ Jean Dujardin (62%)`.

## Astuces

- Les dossiers commençant par `celebrites` peuvent aussi être glissés
  directement dans la zone de dépôt de l'interface.
- Supprimez les dossiers d'exemple si vous ne les utilisez pas.
- Le seuil d'affichage (`CELEB_MAX_DIST` dans `face_reco.py`) évite d'afficher
  des ressemblances trop faibles.
