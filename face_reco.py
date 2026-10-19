"""GiscardVision : reconnaissance faciale webcam, ambiance Giscardpunk/Minitel."""

import json
import os
import shutil
import time
from datetime import date

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_OK = True
except ImportError:
    DND_OK = False

DATA_DIR = "data"
DB_FILE = os.path.join(DATA_DIR, "people.json")
TRAINING_DIR = os.path.join(DATA_DIR, "faces")
CELEB_DIR = "celebrites"
CELEB_DATA_DIR = os.path.join(DATA_DIR, "celeb_model")
IMPORT_DIR = "a_importer"
IMPORT_DONE_DIR = os.path.join(IMPORT_DIR, "traite")
FACE_SIZE = (200, 200)
MIN_FACE_SIZE = (80, 80)
CONFIDENCE_THRESHOLD = 50.0
POLL_INTERVAL = 2.0
PREDICT_INTERVAL = 1.5
SIZE_UPDATE_INTERVAL = 0.5
SIZE_TOLERANCE = 0.12
TRACK_TIMEOUT = 1.0
POS_ALPHA = 0.35
SIZE_ALPHA = 0.6
CELEB_TOP_K = 3
CELEB_MAX_DIST = 90.0
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FONT = cv2.FONT_HERSHEY_SIMPLEX
REF_BOX_HEIGHT = 280.0
BLUR_MIN_VAR = 40.0

CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def draw_text_bg(frame, text, org, font, scale, color, thickness=1):
    """Texte avec fond noir surligne pour un contraste maximal."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)
FIELD_LABELS = [
    ("nom", "Nom"),
    ("prenom", "Prenom"),
    ("age", "Age"),
    ("genre", "Genre (libre : H, F, NB, transfem, ... ce que vous voulez)"),
    ("info", "Info complementaire (facultatif)"),
]


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
    return cv2.resize(gray[y : y + h, x : x + w], FACE_SIZE)


def save_face_sample(gray, box, label):
    os.makedirs(TRAINING_DIR, exist_ok=True)
    samples = [f for f in os.listdir(TRAINING_DIR) if f.startswith(f"{label}_")]
    path = os.path.join(TRAINING_DIR, f"{label}_{len(samples)}.png")
    cv2.imwrite(path, extract_face(gray, box))
    return path


def retrain(db, recognizer):
    images, labels = [], []
    files = sorted(os.listdir(TRAINING_DIR)) if os.path.isdir(TRAINING_DIR) else []
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


def person_by_label(db, label):
    return next((p for p in db.values() if p["label"] == label), None)


def predict_face(gray, box, recognizer):
    if recognizer.empty():
        return None, 0.0
    label, conf = recognizer.predict(extract_face(gray, box))
    confidence = 100 - conf
    if confidence > CONFIDENCE_THRESHOLD:
        return label, confidence
    return None, confidence


class FaceTrack:
    """Visage suivi : position lissee, taille stable (tolerance + maj max 2/s)."""

    _next_id = 0

    def __init__(self, box, now):
        self.id = FaceTrack._next_id
        FaceTrack._next_id += 1
        self.box = [float(v) for v in box]
        self.size_lock_until = now + SIZE_UPDATE_INTERVAL
        self.last_seen = now
        self.person = None
        self.confidence = 0.0
        self.next_predict = 0.0
        self.celeb = None

    def iou(self, box):
        ax, ay, aw, ah = self.box
        bx, by, bw, bh = box
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def update(self, box, now):
        x, y, w, h = box
        self.box[0] += POS_ALPHA * (x - self.box[0])
        self.box[1] += POS_ALPHA * (y - self.box[1])
        if now >= self.size_lock_until and abs(w - self.box[2]) > self.box[2] * SIZE_TOLERANCE:
            self.box[2] += SIZE_ALPHA * (w - self.box[2])
            self.box[3] += SIZE_ALPHA * (h - self.box[3])
            self.size_lock_until = now + SIZE_UPDATE_INTERVAL
        self.last_seen = now

    @property
    def int_box(self):
        return tuple(int(round(v)) for v in self.box)


class FaceTracker:
    def __init__(self):
        self.tracks = []

    def update(self, boxes, now):
        unmatched = []
        for det in boxes:
            best, best_iou = None, 0.2
            for t in self.tracks:
                iou = t.iou(det)
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is not None:
                best.update(det, now)
            else:
                unmatched.append(det)
        for det in unmatched:
            self.tracks.append(FaceTrack(det, now))
        self.tracks = [t for t in self.tracks if now - t.last_seen <= TRACK_TIMEOUT]
        return self.tracks


class Explosion:
    DURATION = 1.5
    N_PART = 70
    COLORS = [(0, 0, 255), (0, 100, 255), (0, 165, 255), (0, 215, 255), (255, 255, 255)]

    def __init__(self, cx, cy, w, now):
        self.t0 = now
        self.cx, self.cy = cx, cy
        self.r = max(20, w // 2)
        rng = np.random.default_rng()
        self.angles = rng.uniform(0, 2 * np.pi, self.N_PART)
        self.speeds = rng.uniform(90, 420, self.N_PART)
        self.sizes = rng.uniform(2, 7, self.N_PART)
        self.color_idx = rng.integers(0, len(self.COLORS), self.N_PART)

    def draw(self, frame, now):
        dt = now - self.t0
        if dt < 0 or dt > self.DURATION:
            return False
        if dt < 0.12:
            cv2.circle(frame, (self.cx, self.cy), int(self.r * (1 + dt * 50)), (255, 255, 255), -1)
        fade = 1 - dt / self.DURATION
        for k in (1, 2, 3):
            r = int(self.r + dt * 300 * k)
            cv2.circle(frame, (self.cx, self.cy), r, (0, int(180 * fade), 255), max(1, int(7 * fade)))
        xs = self.cx + np.cos(self.angles) * self.speeds * dt
        ys = self.cy + np.sin(self.angles) * self.speeds * dt + 60 * dt * dt
        for x, y, s, ci in zip(xs, ys, self.sizes, self.color_idx):
            cv2.circle(frame, (int(x), int(y)), int(s), self.COLORS[ci], -1)
        return True


class CelebMatcher:
    """Compare les visages aux photos de celebrites/ (un sous-dossier par celebrite)."""

    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.names = {}
        self.dirty = True

    def retrain_if_needed(self):
        if not os.path.isdir(CELEB_DIR):
            return False
        subdirs = sorted(d for d in os.listdir(CELEB_DIR)
                         if os.path.isdir(os.path.join(CELEB_DIR, d)))
        images, labels, names = [], [], {}
        for idx, celeb in enumerate(subdirs, start=1):
            folder = os.path.join(CELEB_DIR, celeb)
            count = 0
            for fname in sorted(os.listdir(folder)):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                img = cv2.imread(os.path.join(folder, fname))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)
                if len(boxes):
                    face = extract_face(gray, max(boxes, key=lambda b: b[2] * b[3]))
                    images.append(face)
                    labels.append(idx)
                    count += 1
            if count:
                names[idx] = celeb.replace("_", " ")
        if images:
            self.recognizer.train(images, np.array(labels))
            self.names = names
            self.dirty = False
            return True
        self.names = {}
        self.dirty = False
        return False

    def best_match(self, gray, box):
        if self.dirty:
            self.retrain_if_needed()
        if not self.names or self.recognizer.empty():
            return None, 0.0
        label, dist = self.recognizer.predict(extract_face(gray, box))
        if dist > CELEB_MAX_DIST:
            return None, 0.0
        similarity = max(0.0, 100.0 - dist)
        return self.names.get(label), similarity


class App:
    def __init__(self):
        self.db = load_db()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        retrain(self.db, self.recognizer)
        self.celeb = CelebMatcher()
        self.celeb.retrain_if_needed()
        os.makedirs(IMPORT_DIR, exist_ok=True)

        self.tracker = FaceTracker()
        self.explosions = []
        self.pending_terminate = None
        self.terminate_deadline = 0.0
        self.pending_report = None
        self.report_deadline = 0.0
        self.sur_ecoute = set()
        self.last_poll = 0.0
        self.running = False
        self.source = None
        self.cap = None
        self.open_source(0)
        self.window = "GiscardVision"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 960, 540)
        self.fullscreen = False
        self.gui = Gui(self)
        self.gui.refresh_list()

    # ---- source video ----

    def open_source(self, source):
        new_cap = cv2.VideoCapture(source)
        if not new_cap.isOpened():
            return False
        if self.cap is not None:
            self.cap.release()
        self.cap = new_cap
        self.source = source
        self.tracker = FaceTracker()
        return True

    # ---- enregistrement / import ----

    def prompt_new_person(self, nom_defaut=""):
        values = {}
        for key, label in FIELD_LABELS:
            val = simpledialog.askstring("Nouvelle personne", label + " :",
                                         initialvalue=nom_defaut if key == "nom" else "",
                                         parent=self.gui.root)
            if key == "nom" and not val:
                return None
            values[key] = (val or ("?" if key in ("age", "genre") else "")).strip()
        nom = values["nom"]
        if nom in self.db:
            person = self.db[nom]
        else:
            person = {
                "label": next_label(self.db),
                **values,
                "enregistre_le": date.today().isoformat(),
            }
            self.db[nom] = person
            save_db(self.db)
            self.gui.refresh_list()
        return person

    def edit_person(self, person):
        """Modifie les informations d'une personne enregistree."""
        for key, label in FIELD_LABELS:
            current = str(person.get(key, "") or "")
            val = simpledialog.askstring("Modifier personne", label + " :",
                                         initialvalue=current, parent=self.gui.root)
            if val is None:
                return False
            person[key] = val.strip() or ("?" if key in ("age", "genre") else "")
        old_nom = next(k for k, v in self.db.items() if v is person)
        self.db.pop(old_nom)
        self.db[person["nom"]] = person
        save_db(self.db)
        self.gui.refresh_list()
        self.gui.set_status(f"{person['prenom']} {person['nom']} : informations modifiees.")
        return True

    def add_sample(self, gray, box, person):
        save_face_sample(gray, box, person["label"])
        retrain(self.db, self.recognizer)

    def add_sample_now(self, person):
        """Capture le visage actuellement filme et l'ajoute a la personne selectionnee."""
        ret, frame = self.cap.read()
        if not ret:
            self.gui.set_status("Webcam inaccessible.")
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)
        if len(boxes) != 1:
            self.gui.set_status("Montrez exactement UN visage pour ajouter un echantillon.")
            return
        self.add_sample(gray, boxes[0], person)
        self.gui.set_status(f"Nouvel echantillon pour {person['prenom']} {person['nom']}.")

    def process_folder_for_person(self, folder_path, person):
        added = 0
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)
            if len(boxes):
                biggest = max(boxes, key=lambda b: b[2] * b[3])
                self.add_sample(gray, biggest, person)
                added += 1
        return added

    def process_imports(self):
        moved = []
        os.makedirs(IMPORT_DONE_DIR, exist_ok=True)
        for entry in sorted(os.listdir(IMPORT_DIR)):
            path = os.path.join(IMPORT_DIR, entry)
            if not os.path.isdir(path) or entry == "traite":
                continue
            person = next((p for p in self.db.values() if p["nom"] == entry), None)
            if person is None:
                person = self.prompt_new_person(nom_defaut=entry)
            if person is None:
                continue
            n = self.process_folder_for_person(path, person)
            dest = os.path.join(IMPORT_DONE_DIR, entry)
            os.makedirs(dest, exist_ok=True)
            for fname in os.listdir(path):
                shutil.move(os.path.join(path, fname), os.path.join(dest, fname))
            os.rmdir(path)
            moved.append(f"{person['prenom']} {person['nom']} : {n} echantillon(s)")

        flat = [f for f in os.listdir(IMPORT_DIR) if f.lower().endswith(IMAGE_EXTS)]
        new_persons = 0
        for fname in sorted(flat):
            path = os.path.join(IMPORT_DIR, fname)
            img = cv2.imread(path)
            if img is None:
                shutil.move(path, os.path.join(IMPORT_DONE_DIR, fname))
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=MIN_FACE_SIZE)
            for box in boxes:
                label, _ = predict_face(gray, box, self.recognizer)
                person = person_by_label(self.db, label) if label is not None else None
                if person is None:
                    person = self.prompt_new_person()
                    if person is not None:
                        new_persons += 1
                if person is not None:
                    self.add_sample(gray, box, person)
            shutil.move(path, os.path.join(IMPORT_DONE_DIR, fname))

        if moved or flat:
            self.gui.set_status(" | ".join(moved) if moved
                                else f"Import : {len(flat)} image(s), {new_persons} nouvelle(s) personne(s).")
            self.gui.refresh_list()
        return bool(moved or flat)

    # ---- Terminate / Signalement DGSI ----

    def terminate(self, label):
        self.pending_terminate = label
        self.terminate_deadline = time.time() + 5.0
        self.gui.set_status("Cible verrouillee : montrez son visage a la camera (5 s)...")

    def report(self, label):
        self.pending_report = label
        self.report_deadline = time.time() + 5.0
        self.gui.set_status("Signalement DGSI : montrez son visage a la camera (5 s)...")

    def stop_report(self, label):
        """Retire la cible et la mention sur ecoute pour cette personne."""
        self.sur_ecoute.discard(label)
        person = person_by_label(self.db, label)
        name = f"{person['prenom']} {person['nom']}" if person else "cette personne"
        self.gui.set_status(f"{name} : n'est plus sur ecoute.")

    # ---- boucle video ----

    def draw_track(self, frame, track):
        x, y, w, h = track.int_box
        person, confidence = track.person, track.confidence
        factor = max(0.5, min(2.0, h / REF_BOX_HEIGHT))
        name_s, small_s = 0.48 * factor, 0.4 * factor
        th = max(1, round(1.5 * factor))
        color = (0, 255, 0) if person else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, max(2, th))
        if person:
            line1 = f"{person['prenom']} {person['nom']} | {person['age']} | {person.get('genre', '?')}"
            draw_text_bg(frame, line1, (x, y - 10), FONT, name_s, color, th)
            offset = 0
            if person.get("info"):
                offset += 20
                draw_text_bg(frame, person["info"], (x, y + h + int(offset * factor)), FONT, small_s, (255, 255, 0), 1)
            draw_text_bg(frame, f"factieux a {confidence:.0f}%", (x, y + h + int((offset + 22) * factor)), FONT, small_s, (200, 200, 200), 1)
            if track.celeb is not None:
                name, sim = track.celeb
                draw_text_bg(frame, f"~ {name} ({sim:.0f}%)", (x, y + h + int((offset + 42) * factor)), FONT, small_s, (255, 150, 255), 1)
        else:
            draw_text_bg(frame, "Inconnu", (x, y - 10), FONT, name_s, color, th)
            if track.celeb is not None:
                name, sim = track.celeb
                draw_text_bg(frame, f"~ {name} ({sim:.0f}%)", (x, y + h + int(20 * factor)), FONT, small_s, (255, 150, 255), 1)
        if person is not None and person["label"] in self.sur_ecoute:
            cx = x + w // 2
            cy = y + max(10, int(h * 0.12))
            r = max(14, w // 3)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), max(2, th))
            cv2.circle(frame, (cx, cy), max(5, r // 2), (0, 0, 255), 2)
            cv2.line(frame, (cx - r - 8, cy), (cx + r + 8, cy), (0, 0, 255), 2)
            cv2.line(frame, (cx, cy - r - 8), (cx, cy + r + 8), (0, 0, 255), 2)
            draw_text_bg(frame, "sur ecoute", (x, y + h + int((offset + 62) * factor)), FONT, small_s, (0, 0, 255), max(2, th))

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.gui.set_status("Flux webcam interrompu.")
            self.quit()
            return
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)
            tracks = self.tracker.update(list(boxes), now)
            for track in tracks:
                if now >= track.next_predict:
                    # image floue (mouvement) : on garde la derniere identite connue
                    if cv2.Laplacian(extract_face(gray, track.int_box), cv2.CV_64F).var() >= BLUR_MIN_VAR:
                        label, confidence = predict_face(gray, track.int_box, self.recognizer)
                        track.person = person_by_label(self.db, label) if label is not None else None
                        track.confidence = confidence
                        track.celeb = self.celeb.best_match(gray, track.int_box)
                    track.next_predict = now + PREDICT_INTERVAL
                self.draw_track(frame, track)
                if self.pending_terminate is not None and track.person is not None \
                        and track.person["label"] == self.pending_terminate:
                    x, y, w, h = track.int_box
                    self.explosions.append(Explosion(x + w // 2, y + h // 2, w, now))
                    self.pending_terminate = None
                    self.gui.set_status(f"{track.person['prenom']} {track.person['nom']} : terminee.")
                if self.pending_report is not None and track.person is not None \
                        and track.person["label"] == self.pending_report:
                    self.sur_ecoute.add(track.person["label"])
                    self.pending_report = None
                    self.gui.set_status(f"{track.person['prenom']} {track.person['nom']} : signalee a la DGSI, sur ecoute.")
            if self.pending_terminate is not None and now > self.terminate_deadline:
                self.gui.set_status("Cible non visible a l'ecran.")
                self.pending_terminate = None
            if self.pending_report is not None and now > self.report_deadline:
                self.gui.set_status("Cible non visible a l'ecran.")
                self.pending_report = None
            self.explosions = [e for e in self.explosions if e.draw(frame, now)]

            try:
                win_rect = cv2.getWindowImageRect(self.window)
                win_w, win_h = win_rect[2], win_rect[3]
            except cv2.error:
                win_w, win_h = frame.shape[1], frame.shape[0]
            if win_w > 0 and (win_w != frame.shape[1] or win_h != frame.shape[0]):
                fh, fw = frame.shape[:2]
                scale = min(win_w / fw, win_h / fh)
                new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
                resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                ox, oy = (win_w - new_w) // 2, (win_h - new_h) // 2
                canvas[oy : oy + new_h, ox : ox + new_w] = resized
                cv2.imshow(self.window, canvas)
            else:
                cv2.imshow(self.window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.quit()
                return
            elif key == ord("f"):
                self.fullscreen = not self.fullscreen
                cv2.setWindowProperty(
                    self.window, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord("n"):
                if len(tracks) == 1:
                    track = tracks[0]
                    if track.person is not None:
                        self.add_sample(gray, track.int_box, track.person)
                        self.gui.set_status(f"Nouvel echantillon pour {track.person['prenom']} {track.person['nom']}.")
                    else:
                        person = self.prompt_new_person()
                        if person is not None:
                            self.add_sample(gray, track.int_box, person)
                            self.gui.set_status(f"{person['prenom']} {person['nom']} enregistre !")
                else:
                    self.gui.set_status("Montrez exactement UN visage pour enregistrer.")

        if now - self.last_poll > POLL_INTERVAL:
            self.last_poll = now
            try:
                self.process_imports()
            except Exception as exc:
                self.gui.set_status(f"Erreur import : {exc}")

        if self.running:
            self.gui.root.after(33, self.tick)

    def run(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'acceder a la webcam.")
            if self.cap:
                self.cap.release()
            return
        self.running = True
        self.gui.root.after(33, self.tick)
        self.gui.root.mainloop()

    def quit(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.gui.root.destroy()


class Gui:
    def __init__(self, app):
        self.app = app
        self.root = TkinterDnD.Tk() if DND_OK else tk.Tk()
        self.root.title("GiscardVision - Controle")
        self.root.geometry("470x640")
        self.root.protocol("WM_DELETE_WINDOW", app.quit)
        self.people = []

        # Bandeau tricolore + titre GISCARDVISION (look Minitel)
        strip = tk.Frame(self.root, height=46)
        strip.pack(fill=tk.X)
        strip.pack_propagate(False)
        tk.Frame(strip, bg="#002395").place(relx=0.0, rely=0.0, relwidth=1 / 3, relheight=1.0)
        tk.Frame(strip, bg="#ffffff").place(relx=1 / 3, rely=0.0, relwidth=1 / 3, relheight=1.0)
        tk.Frame(strip, bg="#ED2939").place(relx=2 / 3, rely=0.0, relwidth=1 / 3, relheight=1.0)
        tk.Label(strip, text="GISCARDVISION", font=("Courier", 15, "bold"),
                 bg="#ffffff", fg="#002395").place(relx=0.5, rely=0.5, anchor="center")

        self.status_var = tk.StringVar(value="Pret.")
        top = tk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        drop_txt = ("ZONE DE DEPOT\n\nGlissez ici des images ou des dossiers\n"
                    "(un dossier = une personne, ex : Jean_Dupont/)\n"
                    "Celebrites : deposez-les dans le dossier celebrites/")
        self.drop_zone = tk.Label(top, text=drop_txt, relief="groove", bd=2, height=6, bg="#f0f0f0")
        self.drop_zone.pack(fill=tk.X, pady=5)
        if DND_OK:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.bind("<<Drop>>", self.on_drop)
            self.drop_zone.bind("<Enter>", lambda e: self.drop_zone.config(bg="#dbeaff"))
            self.drop_zone.bind("<Leave>", lambda e: self.drop_zone.config(bg="#f0f0f0"))
        else:
            tk.Button(top, text="Importer des images... (pip install tkinterdnd2 pour le glisser-deposer)",
                      command=self.browse_files).pack(pady=5)

        tk.Label(top, text="Personnes enregistrees :").pack(anchor="w", pady=(5, 0))
        self.listbox = tk.Listbox(top, height=8)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        self.details = tk.Label(top, text="Selectionnez une personne.", justify=tk.LEFT, anchor="w", wraplength=440)
        self.details.pack(fill=tk.X, pady=5)

        self.edit_btn = tk.Button(top, text="Modifier les informations...", state=tk.DISABLED,
                                  command=self.on_edit)
        self.edit_btn.pack(fill=tk.X, pady=2)

        self.add_btn = tk.Button(top, text="Ajouter un échantillon (photo)", state=tk.DISABLED,
                                 command=self.on_add_sample)
        self.add_btn.pack(fill=tk.X, pady=2)

        self.terminate_btn = tk.Button(
            top, text="TERMINATE", state=tk.DISABLED, bg="#ED2939", fg="white",
            font=("Courier", 12, "bold"), command=self.on_terminate,
        )
        self.terminate_btn.pack(fill=tk.X, pady=5)

        self.dgsi_btn = tk.Button(
            top, text="Signaler à la DGSI", state=tk.DISABLED, bg="#002395", fg="white",
            font=("Courier", 12, "bold"), command=self.on_report,
        )
        self.dgsi_btn.pack(fill=tk.X, pady=5)

        self.stop_btn = tk.Button(
            top, text="Cesser d'écouter", state=tk.DISABLED, bg="#ffffff", fg="#002395",
            font=("Courier", 12, "bold"), command=self.on_stop_report,
        )
        self.stop_btn.pack(fill=tk.X, pady=5)

        tk.Label(self.root, textvariable=self.status_var, fg="#0b57d0", wraplength=450, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=5)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        self.people = sorted(self.app.db.values(), key=lambda p: p["nom"].lower())
        for p in self.people:
            self.listbox.insert(tk.END, f"{p['prenom']} {p['nom']}")
        self.terminate_btn.config(state=tk.DISABLED)
        self.dgsi_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.edit_btn.config(state=tk.DISABLED)
        self.add_btn.config(state=tk.DISABLED)
        self.details.config(text="Selectionnez une personne.")

    def on_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        p = self.people[sel[0]]
        self.details.config(
            text=f"Nom : {p['nom']}\nPrenom : {p['prenom']}\nAge : {p['age']}  |  Genre : {p.get('genre', '?')}\n"
                 f"Info : {p.get('info', '')}\nEnregistre le : {p.get('enregistre_le', '?')}"
        )
        self.terminate_btn.config(state=tk.NORMAL)
        self.dgsi_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)
        self.edit_btn.config(state=tk.NORMAL)
        self.add_btn.config(state=tk.NORMAL)

    def on_edit(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.edit_person(self.people[sel[0]])

    def on_add_sample(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.add_sample_now(self.people[sel[0]])

    def on_report(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.report(self.people[sel[0]]["label"])

    def on_stop_report(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.stop_report(self.people[sel[0]]["label"])

    def on_terminate(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.terminate(self.people[sel[0]]["label"])

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        os.makedirs(IMPORT_DIR, exist_ok=True)
        n = 0
        for path in paths:
            base = os.path.basename(os.path.normpath(path))
            if os.path.isdir(path):
                if base.lower().startswith("celebrite"):
                    dest = os.path.join(CELEB_DIR, base)
                    os.makedirs(CELEB_DIR, exist_ok=True)
                    shutil.copytree(path, dest, dirs_exist_ok=True)
                    self.app.celeb.dirty = True
                    self.app.celeb.retrain_if_needed()
                    self.set_status(f"Celebites importees depuis {base} : modele recalcule.")
                else:
                    shutil.copytree(path, os.path.join(IMPORT_DIR, base), dirs_exist_ok=True)
                n += 1
            elif os.path.isfile(path) and path.lower().endswith(IMAGE_EXTS):
                shutil.copy2(path, os.path.join(IMPORT_DIR, base))
                n += 1
        if n and any(not os.path.basename(os.path.normpath(p)).lower().startswith("celebrite")
                     for p in paths if os.path.isdir(p)):
            self.set_status(f"{n} element(s) recupere(s), traitement...")
            self.root.after(100, self.app.process_imports)

    def browse_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        for path in files:
            shutil.copy2(path, os.path.join(IMPORT_DIR, os.path.basename(path)))
        if files:
            self.set_status("Import en cours...")
            self.root.after(100, self.app.process_imports)

    def set_status(self, msg):
        self.status_var.set(msg)


if __name__ == "__main__":
    app = App()
    app.run()
