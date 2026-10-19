"""Reconnaissance faciale webcam : detection + infos (nom, prenom, age, sexe)."""

import json
import os
import shutil
import time
from datetime import date

import cv2
import numpy as np

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "people.json")
TRAINING_DIR = os.path.join(DATA_DIR, "faces")
IMPORT_DIR = "a_importer"
IMPORT_DONE_DIR = os.path.join(IMPORT_DIR, "traite")
FACE_SIZE = (200, 200)
MIN_FACE_SIZE = (80, 80)
CONFIDENCE_THRESHOLD = 50.0
REF_BOX_HEIGHT = 280.0
POLL_INTERVAL = 2.0
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FONT = cv2.FONT_HERSHEY_SIMPLEX

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
    samples = [f for f in os.listdir(TRAINING_DIR) if f.startswith(f"{label}_")]
    path = os.path.join(TRAINING_DIR, f"{label}_{len(samples)}.png")
    cv2.imwrite(path, extract_face(gray, box))
    return path


def retrain(db, recognizer):
    images, labels = [], []
    if os.path.isdir(TRAINING_DIR):
        files = sorted(os.listdir(TRAINING_DIR))
    else:
        files = []
    for person in db.values():
        prefix = f"{person['label']}_"
        for fname in files:
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


def predict_face(gray, box, recognizer):
    if recognizer.empty():
        return None, 0.0
    label, conf = recognizer.predict(extract_face(gray, box))
    confidence = 100 - conf
    if confidence > CONFIDENCE_THRESHOLD:
        return label, confidence
    return None, confidence


def person_by_label(db, label):
    return next((p for p in db.values() if p["label"] == label), None)


def create_person(db, gray, box):
    nom = input("Nom : ").strip()
    if not nom:
        return None
    prenom = input("Prenom : ").strip()
    age = input("Age : ").strip()
    sexe = input("Sexe (H/F) : ").strip()
    extra = input("Info complementaire (facultatif) : ").strip()
    if nom in db:
        person = db[nom]
    else:
        person = {
            "label": next_label(db),
            "nom": nom,
            "prenom": prenom,
            "age": age,
            "sexe": sexe,
            "info": extra,
            "enregistre_le": date.today().isoformat(),
        }
        db[nom] = person
    save_face_sample(gray, box, person["label"])
    save_db(db)
    return person


def register_from_webcam(gray, box, db, recognizer):
    label, _ = predict_face(gray, box, recognizer)
    person = person_by_label(db, label) if label is not None else None
    if person:
        save_face_sample(gray, box, person["label"])
        retrain(db, recognizer)
        return f"Nouvel echantillon pour {person['prenom']} {person['nom']}."
    person = create_person(db, gray, box)
    if person is None:
        return "Enregistrement annule."
    retrain(db, recognizer)
    return f"{person['prenom']} {person['nom']} enregistre !"


def process_imported_images(db, recognizer):
    """Traite les images deposees dans a_importer/ (plusieurs par personne OK)."""
    if not os.path.isdir(IMPORT_DIR):
        return None
    files = [f for f in os.listdir(IMPORT_DIR) if f.lower().endswith(IMAGE_EXTS)]
    if not files:
        return None
    os.makedirs(IMPORT_DONE_DIR, exist_ok=True)
    nb_ajoutes, nb_visages, inconnus = 0, 0, 0
    for fname in sorted(files):
        path = os.path.join(IMPORT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            shutil.move(path, os.path.join(IMPORT_DONE_DIR, fname))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)
        for box in boxes:
            nb_visages += 1
            label, _ = predict_face(gray, box, recognizer)
            person = person_by_label(db, label) if label is not None else None
            if person is None:
                print(f"\nVisage inconnu dans {fname}.")
                person = create_person(db, gray, box)
                if person is not None:
                    inconnus += 1
                    retrain(db, recognizer)
            else:
                save_face_sample(gray, box, person["label"])
                retrain(db, recognizer)
            if person is not None:
                nb_ajoutes += 1
        shutil.move(path, os.path.join(IMPORT_DONE_DIR, fname))
    if nb_visages == 0:
        return f"{len(files)} image(s) importee(s), aucun visage detecte."
    return f"Import : {nb_ajoutes} visage(s) ajoutes ({inconnus} nouvelle(s) personne(s))."


def text_scales(h):
    """Echelles proportionnelles a la hauteur du cadre du visage."""
    factor = max(0.5, min(2.0, h / REF_BOX_HEIGHT))
    name = 0.48 * factor
    small = 0.4 * factor
    thickness = max(1, round(1.5 * factor))
    return name, small, thickness


def draw_info(frame, box, person, confidence):
    x, y, w, h = box
    name_s, small_s, th = text_scales(h)
    color = (0, 255, 0) if person else (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, max(2, th))
    if person:
        line1 = f"{person['prenom']} {person['nom']} | {person['age']} ans | {person['sexe']}"
        cv2.putText(frame, line1, (x, y - 10), FONT, name_s, color, th)
        if person.get("info"):
            cv2.putText(frame, person["info"], (x, y + h + round(20 * name_s / 0.48)), FONT, small_s, (255, 255, 0), 1)
        cv2.putText(frame, f"confiance: {confidence:.0f}", (x, y + h + round(40 * name_s / 0.48)), FONT, small_s, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Inconnu", (x, y - 10), FONT, name_s, color, th)


def main():
    db = load_db()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    retrain(db, recognizer)
    os.makedirs(IMPORT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : impossible d'acceder a la webcam.")
        return

    print("Webcam activee. Touches : N = nouvel enregistrement | F = plein ecran | Q = quitter")
    print(f"Deposez des images dans ./{IMPORT_DIR}/ pour les importer (traitement automatique).")
    window = "Reconnaissance faciale"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    fullscreen = False
    message = ""
    last_poll = 0.0
    message_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)

        for box in boxes:
            label, confidence = predict_face(gray, box, recognizer)
            person = person_by_label(db, label) if label is not None else None
            draw_info(frame, box, person, confidence)

        now = time.time()
        if now - last_poll > POLL_INTERVAL:
            last_poll = now
            msg = process_imported_images(db, recognizer)
            if msg:
                message = msg
                message_time = now
        if message and now - message_time < 5.0:
            cv2.putText(frame, message, (10, 30), FONT, 0.56, (0, 200, 255), 1)
        elif message and now - message_time >= 5.0:
            message = ""

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
                message = register_from_webcam(gray, boxes[0], db, recognizer)
                message_time = time.time()
            else:
                message = "Montrez exactement UN visage pour enregistrer."
                message_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
