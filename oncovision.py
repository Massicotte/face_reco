"""ONCOVISION - Depistage national des cancers de la societe (spectacle, satire).

Detection des visages en direct sur le plateau ; l'operateur affecte un
"type de cancer" satirique a chaque comedien, puis le bouton ELIMINER LE CANCER
floute/pixelise la personne en direct (elle disparait de l'ecran, mais reste
la, juste cachee).
"""

import tkinter as tk
from tkinter import simpledialog, messagebox
import time

import cv2
import numpy as np

FACE_SIZE = (200, 200)
MIN_FACE_SIZE = (70, 70)
SIZE_UPDATE_INTERVAL = 0.5
SIZE_TOLERANCE = 0.12
TRACK_TIMEOUT = 1.0
POS_ALPHA = 0.35
SIZE_ALPHA = 0.6
MAX_SLOTS = 6
PIXEL_STRONG = 18          # taille des pixels du floutage "disparition"
CANCER_TYPES = [
    "ECOLOGAUCHIASSE A TOTEBAG",
    "WOKISTE PIPOU SAFE DE TWITTER",
    "GAUCHISTE A BARBE NON NEGOCIABLE",
    "BOBO VELO-CHANGEANT",
    "SYNDICALISTE CHRONIQUE",
    "INTELLO DE CINE-CLUB",
    "ANTIFA DE TERRASSE",
    "GAUCHE MORALE A SANDALES",
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def draw_text_bg(frame, text, org, scale, color, thickness=1):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + base + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)


def pixelate_region(frame, x, y, w, h, pixel=PIXEL_STRONG):
    """Floute fortement une region : la personne disparait sans etre effacee."""
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // pixel), max(1, (y2 - y1) // pixel)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


class FaceTrack:
    _next_id = 1

    def __init__(self, box, now):
        self.id = FaceTrack._next_id
        FaceTrack._next_id += 1
        self.box = [float(v) for v in box]
        self.size_lock_until = now + SIZE_UPDATE_INTERVAL
        self.last_seen = now
        self.cancer = None
        self.eliminated = False

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


class App:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.window = "ONCOVISION - Depistage National"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 960, 540)
        self.tracker_tracks = []
        self.running = False
        self.gui = Gui(self)

    def assign_cancer(self, track_id):
        track = next((t for t in self.tracker_tracks if t.id == track_id), None)
        if track is None:
            return
        menu = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(CANCER_TYPES))
        choix = simpledialog.askstring(
            "ONCOVISION", f"Type de cancer pour le visage #{track_id} :\n\n{menu}\n\n"
                          "Numero (1-8) ou nom libre :",
            parent=self.gui.root)
        if choix is None:
            return
        choix = choix.strip()
        if choix.isdigit() and 1 <= int(choix) <= len(CANCER_TYPES):
            track.cancer = CANCER_TYPES[int(choix) - 1]
        elif choix:
            track.cancer = choix.upper()
        self.gui.refresh_slots()

    def eliminate(self, track_id):
        track = next((t for t in self.tracker_tracks if t.id == track_id), None)
        if track is None:
            return
        track.eliminated = True
        self.gui.refresh_slots()

    def restore(self, track_id):
        track = next((t for t in self.tracker_tracks if t.id == track_id), None)
        if track is None:
            return
        track.eliminated = False
        self.gui.refresh_slots()

    def tick(self):
        ret, frame = self.cap.read()
        now = time.time()
        if not ret:
            self.quit()
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=MIN_FACE_SIZE)

        unmatched = []
        for det in boxes:
            best, best_iou = None, 0.2
            for t in self.tracker_tracks:
                iou = t.iou(det)
                if iou > best_iou:
                    best, best_iou = t, iou
            if best is not None:
                best.update(det, now)
            else:
                unmatched.append(det)
        for det in unmatched:
            self.tracker_tracks.append(FaceTrack(det, now))
        self.tracker_tracks = [t for t in self.tracker_tracks if now - t.last_seen <= TRACK_TIMEOUT]

        for t in self.tracker_tracks:
            x, y, w, h = t.int_box
            if t.eliminated:
                pixelate_region(frame, x, y, w, h)
                color = (255, 30, 20)
                draw_text_bg(frame, f"#{t.id} CANCER ELIMINE", (x, y - 10), 0.55, color, 1)
            elif t.cancer:
                color = (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_text_bg(frame, f"#{t.id} {t.cancer}", (x, y - 10), 0.5, color, 1)
            else:
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_text_bg(frame, f"#{t.id} A DEPISTER", (x, y - 10), 0.5, color, 1)

        # bandeau tricolore ONCOVISION
        bh = min(46, frame.shape[0])
        third = frame.shape[1] // 3
        frame[:bh, :third] = (255, 30, 20)
        frame[:bh, third:2 * third] = (240, 240, 240)
        frame[:bh, 2 * third:] = (40, 30, 215)
        draw_text_bg(frame, "O N C O V I S I O N", (third, 22), 0.62, (0, 0, 0), 2)
        cv2.imshow(self.window, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.quit()
            return
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
    """Pupitre de l'operateur : un slot par visage detecte."""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("ONCOVISION - Pupitre de depistage")
        self.root.geometry("560x720")
        self.root.configure(bg="#060628")
        self.root.protocol("WM_DELETE_WINDOW", app.quit)
        BG = "#060628"

        for col in ("#002395", "#ffffff", "#ED2939"):
            tk.Label(self.root, bg=col, height=1).pack(fill=tk.X)
        tk.Label(self.root, text="O N C O V I S I O N", bg=BG, fg="#58d6ff",
                 font=("Courier", 16, "bold")).pack(pady=(4, 0))
        tk.Label(self.root, text="DEPISTAGE NATIONAL DES CANCERS DE LA SOCIETE",
                 bg=BG, fg="#ffb000", font=("Courier", 9, "bold")).pack()
        tk.Label(self.root, text="MINISTERE DE LA SANTE SOCIALE - EDITION 2028 - SECRET DEFENSE",
                 bg=BG, fg="#7a7a8a", font=("Courier", 7)).pack(pady=(0, 4))

        self.slots_frame = tk.Frame(self.root, bg=BG)
        self.slots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        tk.Label(self.root, bg=BG, fg="#7a7a8a", font=("Courier", 8),
                 text="Q (sur la video) : quitter | Un slot par visage detecte en direct"
                 ).pack(pady=(0, 4))

    def refresh_slots(self):
        for w in self.slots_frame.winfo_children():
            w.destroy()
        tracks = self.app.tracker_tracks[:MAX_SLOTS]
        if not tracks:
            tk.Label(self.slots_frame, text="Aucun visage detecte.\nAttendez une entree en scene...",
                     bg="#060628", fg="#7a7a8a", font=("Courier", 10)).pack(pady=30)
        for t in tracks:
            slot = tk.Frame(self.slots_frame, bg="#0b0b3a", relief="groove", bd=1)
            slot.pack(fill=tk.X, pady=4, ipady=3)
            label = f"#{t.id} - {t.cancer or 'a depister'}"
            color = "#00ff9d" if not t.eliminated else "#ED2939"
            tk.Label(slot, text=label, bg="#0b0b3a", fg=color,
                     font=("Courier", 10, "bold"), anchor="w").pack(side=tk.LEFT, padx=6)
            tk.Button(slot, text="TYPE DE CANCER",
                      font=("Courier", 8, "bold"), bg="#f7d117", fg="#060628",
                      command=lambda i=t.id: self.app.assign_cancer(i)).pack(side=tk.LEFT, padx=3)
            if t.eliminated:
                tk.Button(slot, text="RESTAURER (rechute)",
                          font=("Courier", 8, "bold"), bg="#00b04f", fg="#060628",
                          command=lambda i=t.id: self.app.restore(i)).pack(side=tk.LEFT, padx=3)
            else:
                tk.Button(slot, text="ELIMINER LE CANCER",
                          font=("Courier", 8, "bold"), bg="#ED2939", fg="white",
                          command=lambda i=t.id: self.app.eliminate(i)).pack(side=tk.LEFT, padx=3)

    def tick_gui(self):
        self.refresh_slots()
        if self.app.running:
            self.root.after(500, self.tick_gui)


if __name__ == "__main__":
    app = App()
    app.gui.tick_gui()
    app.run()
