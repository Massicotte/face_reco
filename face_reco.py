"""Reconnaissance faciale webcam : detection + infos (nom, prenom, age, sexe)."""

import json
import os
from datetime import date

import cv2
import numpy as np

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "people.json")
TRAINING_DIR = os.path.join(DATA_DIR, "faces")
FACE_SIZE = (200, 200)
MIN_FACE_SIZE = (80, 80)
CONFIDENCE_THRESHOLD = 50.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE_NAME = 0.48
SCALE_INFO = 0.4
SCALE_CONF = 0.4
SCALE_MSG = 0.56

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_db(db):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def extract_face(gray, box):
    x, y, w, h = box
    face = gray[y : y + h, x : x + w]
    return cv2.resize(face, FACE_SIZE)


def save_face_sample(gray, box, label):
    os.makedirs(TRAINING_DIR, exist_ok=True)
    samples = [
        f for f in os.listdir(TRAINING_DIR) if f.startswith(f"{label}_")
    ]
    index = len(samples)
    path = os.path.join(TRAINING_DIR, f"{label}_{index}.png")
    cv2.imwrite(path, extract_face(gray, box))
    return path


def retrain(db, recognizer):
    images, labels = [], []
    for person in db.values():
        prefix = f"{person['label']}_"
        for fname in os.listdir(TRAINING_DIR) if os.path.isdir(TRAINING_DIR) else []:
            if fname.startswith(prefix):
                img = cv2.imread(os.path.join(TRAINING_DIR, fname), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, FACE_SIZE))
                    labels.append(person["label"])
    if images:
        recognizer.train(images, np.array(labels))
        return True
    return False


def next_label(db):
    return max((p["label"] for p in db.values()), default=0) + 1


def prompt_person_info(nom_existant=None):
    if nom_existant:
        print(f"Ajout d'un echantillon pour {nom_existant} existant.")
        return nom_existant, None
    nom = input("Nom : ").strip()
    prenom = input("Prenom : ").strip()
    age = input("Age : ").strip()
    sexe = input("Sexe (H/F) : ").strip()
    extra = input("Info complementaire (facultatif) : ").strip()
    return nom, {
        "nom": nom,
        "prenom": prenom,
        "age": age,
        "sexe": sexe,
        "info": extra,
        "enregistre_le": date.today().isoformat(),
    }


def register_person(gray, box, db, recognizer):
    face = extract_face(gray, box)
    label, conf = recognizer.predict(face) if not recognizer.empty() else (0, 100)
    confidence = 100 - conf
    known = None
    if confidence > CONFIDENCE_THRESHOLD:
        known = next((p for p in db.values() if p["label"] == label), None)

    nom, infos = prompt_person_info(known["nom"] if known else None)
    if not nom:
        return "Enregistrement annule."

    if known:
        person = known
    elif nom in db:
        person = db[nom]
    else:
        person = {"label": next_label(db), **infos}
        db[nom] = person
    save_face_sample(gray, box, person["label"])
    save_db(db)
    retrain(db, recognizer)
    return f"{person.get('prenom', '')} {nom} : nouvel echantillon enregistre."


def recognize(gray, box, db, recognizer):
    if recognizer.empty() or not db:
        return None, 0.0
    label, conf = recognizer.predict(extract_face(gray, box))
    confidence = 100 - conf
    if confidence > CONFIDENCE_THRESHOLD:
        person = next((p for p in db.values() if p["label"] == label), None)
        return person, confidence
    return None, confidence


def draw_info(frame, box, person, confidence):
    x, y, w, h = box
    color = (0, 255, 0) if person else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if person:
        line1 = f"{person['prenom']} {person['nom']} | {person['age']} ans | {person['sexe']}"
        cv2.putText(frame, line1, (x, y - 10), FONT, SCALE_NAME, color, 1)
        if person.get("info"):
            cv2.putText(frame, person["info"], (x, y + h + 20), FONT, SCALE_INFO, (255, 255, 0), 1)
        cv2.putText(frame, f"confiance: {confidence:.0f}", (x, y + h + 40), FONT, SCALE_CONF, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Inconnu", (x, y - 10), FONT, SCALE_NAME, color, 1)


def main():
    db = load_db()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    retrain(db, recognizer)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'acceder a la webcam.")
        return

    print("Webcam activee. Touches : N = nouvel enregistrement | F = plein ecran | Q = quitter")
    window = "Reconnaissance faciale"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    fullscreen = False
    message = ""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)

        for box in boxes:
            person, confidence = recognize(gray, box, db, recognizer)
            draw_info(frame, box, person, confidence)

        if message:
            cv2.putText(frame, message, (10, 30), FONT, SCALE_MSG, (0, 200, 255), 1)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                window, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
            )
        elif key == ord("n"):
            if len(boxes) == 1:
                message = register_person(gray, boxes[0], db, recognizer)
            else:
                message = "Montrez exactement UN visage pour enregistrer."

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
