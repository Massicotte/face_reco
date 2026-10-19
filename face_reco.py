"""Reconnaissance faciale webcam : detection + infos + interface graphique."""

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
IMPORT_DIR = "a_importer"
IMPORT_DONE_DIR = os.path.join(IMPORT_DIR, "traite")
FACE_SIZE = (200, 200)
MIN_FACE_SIZE = (80, 80)
CONFIDENCE_THRESHOLD = 50.0
POLL_INTERVAL = 2.0
PREDICT_INTERVAL = 0.5
SIZE_UPDATE_INTERVAL = 0.5
SIZE_TOLERANCE = 0.12
TRACK_TIMEOUT = 1.0
POS_ALPHA = 0.35
SIZE_ALPHA = 0.6
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
FONT = cv2.FONT_HERSHEY_SIMPLEX
REF_BOX_HEIGHT = 280.0

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
    """Visage suivi : position lissee, taille stable (tolerance + mise a jour max 2/s)."""

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


class App:
    def __init__(self):
        self.db = load_db()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        retrain(self.db, self.recognizer)
        os.makedirs(IMPORT_DIR, exist_ok=True)

        self.tracker = FaceTracker()
        self.explosions = []
        self.pending_terminate = None
        self.terminate_deadline = 0.0
        self.last_poll = 0.0
        self.running = False
        self.cap = cv2.VideoCapture(0)
        self.window = "Reconnaissance faciale"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        self.fullscreen = False
        self.gui = Gui(self)
        self.gui.refresh_list()

    # ---- enregistrement / import ----

    def prompt_new_person(self, nom_defaut=""):
        nom = simpledialog.askstring("Nouvelle personne", "Nom :", initialvalue=nom_defaut, parent=self.gui.root)
        if not nom:
            return None
        prenom = simpledialog.askstring("Nouvelle personne", f"Prenom de {nom} :", parent=self.gui.root) or ""
        age = simpledialog.askstring("Nouvelle personne", f"Age de {nom} :", parent=self.gui.root) or "?"
        sexe = simpledialog.askstring("Nouvelle personne", f"Sexe de {nom} (H/F) :", parent=self.gui.root) or "?"
        extra = simpledialog.askstring("Nouvelle personne", "Info complementaire (facultatif) :", parent=self.gui.root) or ""
        if nom in self.db:
            person = self.db[nom]
        else:
            person = {
                "label": next_label(self.db),
                "nom": nom,
                "prenom": prenom,
                "age": age,
                "sexe": sexe,
                "info": extra,
                "enregistre_le": date.today().isoformat(),
            }
            self.db[nom] = person
            save_db(self.db)
            self.gui.refresh_list()
        return person

    def add_sample(self, gray, box, person):
        save_face_sample(gray, box, person["label"])
        retrain(self.db, self.recognizer)

    def process_folder_for_person(self, folder_path, person):
        """Toutes les images d'un sous-dossier -> cette personne (plus grand visage par photo)."""
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
        """Sous-dossiers de a_importer/ = une personne ; images a la racine = reconnaissance."""
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
            self.gui.set_status(" | ".join(moved) if moved else f"Import : {len(flat)} image(s), {new_persons} nouvelle(s) personne(s).")
            self.gui.refresh_list()
        return bool(moved or flat)

    # ---- Terminate ----

    def terminate(self, label):
        self.pending_terminate = label
        self.terminate_deadline = time.time() + 5.0
        self.gui.set_status("Cible verrouillee : montrez son visage a la camera (5 s)...")

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
            line1 = f"{person['prenom']} {person['nom']} | {person['age']} ans | {person['sexe']}"
            cv2.putText(frame, line1, (x, y - 10), FONT, name_s, color, th)
            if person.get("info"):
                cv2.putText(frame, person["info"], (x, y + h + int(20 * factor)), FONT, small_s, (255, 255, 0), 1)
            cv2.putText(frame, f"confiance: {confidence:.0f}", (x, y + h + int(42 * factor)), FONT, small_s, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Inconnu", (x, y - 10), FONT, name_s, color, th)

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)
            tracks = self.tracker.update(list(boxes), now)
            for track in tracks:
                if now >= track.next_predict:
                    label, confidence = predict_face(gray, track.int_box, self.recognizer)
                    track.person = person_by_label(self.db, label) if label is not None else None
                    track.confidence = confidence
                    track.next_predict = now + PREDICT_INTERVAL
                self.draw_track(frame, track)
                if self.pending_terminate is not None and track.person is not None \
                        and track.person["label"] == self.pending_terminate:
                    x, y, w, h = track.int_box
                    self.explosions.append(Explosion(x + w // 2, y + h // 2, w, now))
                    self.pending_terminate = None
                    self.gui.set_status(f"{track.person['prenom']} {track.person['nom']} : terminee.")
            if self.pending_terminate is not None and now > self.terminate_deadline:
                self.gui.set_status("Cible non visible a l'ecran.")
                self.pending_terminate = None
            self.explosions = [e for e in self.explosions if e.draw(frame, now)]
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
        if not self.cap.isOpened():
            messagebox.showerror("Erreur", "Impossible d'acceder a la webcam.")
            self.cap.release()
            return
        self.running = True
        self.gui.root.after(33, self.tick)
        self.gui.root.mainloop()

    def quit(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.gui.root.destroy()


class Gui:
    def __init__(self, app):
        self.app = app
        self.root = TkinterDnD.Tk() if DND_OK else tk.Tk()
        self.root.title("Face Reco - Controle")
        self.root.geometry("460x560")
        self.root.protocol("WM_DELETE_WINDOW", app.quit)
        self.people = []

        self.status_var = tk.StringVar(value="Pret.")
        top = tk.Frame(self.root)
        top.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        drop_txt = ("ZONE DE DEPOT\n\nGlissez ici des images ou des dossiers\n"
                    "(un dossier = une personne, ex : Jean_Dupont/)")
        self.drop_zone = tk.Label(top, text=drop_txt, relief="groove", bd=2, height=5, bg="#f0f0f0")
        self.drop_zone.pack(fill=tk.X)
        if DND_OK:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.bind("<<Drop>>", self.on_drop)
            self.drop_zone.bind("<Enter>", lambda e: self.drop_zone.config(bg="#dbeaff"))
            self.drop_zone.bind("<Leave>", lambda e: self.drop_zone.config(bg="#f0f0f0"))
        else:
            tk.Button(top, text="Importer des images... (glisser-deposer indisponible : pip install tkinterdnd2)",
                      command=self.browse_files).pack(pady=5)

        tk.Label(top, text="Personnes enregistrees :").pack(anchor="w", pady=(10, 0))
        self.listbox = tk.Listbox(top, height=10)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        self.details = tk.Label(top, text="Selectionnez une personne.", justify=tk.LEFT, anchor="w", wraplength=430)
        self.details.pack(fill=tk.X, pady=5)

        self.terminate_btn = tk.Button(
            top, text="TERMINATE", state=tk.DISABLED, bg="#d93025", fg="white",
            font=("Arial", 12, "bold"), command=self.on_terminate,
        )
        self.terminate_btn.pack(fill=tk.X, pady=5)

        tk.Label(self.root, textvariable=self.status_var, fg="#0b57d0", wraplength=440, justify=tk.LEFT).pack(
            fill=tk.X, padx=10, pady=5)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        self.people = sorted(self.app.db.values(), key=lambda p: p["nom"].lower())
        for p in self.people:
            self.listbox.insert(tk.END, f"{p['prenom']} {p['nom']}")
        self.terminate_btn.config(state=tk.DISABLED)
        self.details.config(text="Selectionnez une personne.")

    def on_select(self, _event):
        sel = self.listbox.curselection()
        if not sel:
            return
        p = self.people[sel[0]]
        self.details.config(
            text=f"Nom : {p['nom']}\nPrenom : {p['prenom']}\nAge : {p['age']}  |  Sexe : {p['sexe']}\n"
                 f"Info : {p.get('info', '')}\nEnregistre le : {p.get('enregistre_le', '?')}"
        )
        self.terminate_btn.config(state=tk.NORMAL)

    def on_terminate(self):
        sel = self.listbox.curselection()
        if sel:
            self.app.terminate(self.people[sel[0]]["label"])

    def on_drop(self, event):
        paths = self.root.tk.splitlist(event.data)
        os.makedirs(IMPORT_DIR, exist_ok=True)
        n = 0
        for path in paths:
            if os.path.isdir(path):
                dest = os.path.join(IMPORT_DIR, os.path.basename(os.path.normpath(path)))
                shutil.copytree(path, dest, dirs_exist_ok=True)
                n += 1
            elif os.path.isfile(path) and path.lower().endswith(IMAGE_EXTS):
                shutil.copy2(path, os.path.join(IMPORT_DIR, os.path.basename(path)))
                n += 1
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
