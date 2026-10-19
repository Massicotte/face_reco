import json
import os
from datetime import date

import cv2
import numpy as np

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "people.json")
TRAINING_DIR = os.path.join(DATA_DIR, "faces")
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_db(db):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def get_label(db, name):
    labels = {p["label"]: p for p in db.values()}
    next_label = max((l for l in labels), default=0) + 1
    return next_label


def register_new_person(frame, faces, db, recognizer):
    if len(faces) != 1:
        return "Montrez exactement UN visage à la caméra pour enregistrer."
    nom = input("Nom : ").strip()
    prenom = input("Prénom : ").strip()
    age = input("Âge : ").strip()
    sexe = input("Sexe (H/F) : ").strip()
    extra = input("Info complémentaire (facultatif) : ").strip()

    x, y, w, h = faces[0]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = gray[y : y + h, x : x + w]
    face = cv2.resize(face, (200, 200))

    os.makedirs(TRAINING_DIR, exist_ok=True)
    label = get_label(db, nom)
    cv2.imwrite(os.path.join(TRAINING_DIR, f"{label}.png"), face)

    db[nom] = {
        "label": label,
        "nom": nom,
        "prenom": prenom,
        "age": age,
        "sexe": sexe,
        "info": extra,
        "enregistre_le": date.today().isoformat(),
    }
    save_db(db)
    retrain(db, recognizer)
    return f"{prenom} {nom} enregistré !"


def retrain(db, recognizer):
    images, labels = [], []
    for person in db.values():
        path = os.path.join(TRAINING_DIR, f"{person['label']}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(person["label"])
    if images:
        recognizer.train(images, np.array(labels))


def draw_info(frame, x, y, w, h, person, confidence):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if person:
        text = f"{person['prenom']} {person['nom']} | {person['age']} ans | {person['sexe']}"
        cv2.putText(frame, text, (x, y - 10), FONT, 0.6, (0, 255, 0), 2)
        if person.get("info"):
            cv2.putText(frame, person["info"], (x, y + h + 20), FONT, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"confiance: {confidence:.0f}", (x, y + h + 40), FONT, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Inconnu", (x, y - 10), FONT, 0.6, (0, 0, 255), 2)


def main():
    db = load_db()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    retrain(db, recognizer)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'accéder à la webcam.")
        return

    print("Webcam activee. Touches : N = nouvel enregistrement | Q = quitter")
    message = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        for x, y, w, h in faces:
            person, confidence = None, 0.0
            if len(db) > 0:
                face = gray[y : y + h, x : x + w]
                face = cv2.resize(face, (200, 200))
                label, conf = recognizer.predict(face)
                confidence = 100 - conf
                if confidence > 50:
                    person = next((p for p in db.values() if p["label"] == label), None)
            draw_info(frame, x, y, w, h, person, confidence)

        if message:
            cv2.putText(frame, message, (10, 30), FONT, 0.7, (0, 200, 255), 2)

        cv2.imshow("Reconnaissance faciale - Q pour quitter, N pour ajouter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            message = register_new_person(frame, list(faces), db, recognizer)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
