import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import librosa
import matplotlib
import numpy as np
import soundfile as sf

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

FMIN_ANALISIS = 20.0
FMAX_ANALISIS = 20000.0
SUAVIZADO_OCT = 1.0 / 3.0

def recortar_silencios(y, sr):
    if len(y) == 0:
        return y
    try:
        _, idx = librosa.effects.trim(y.astype(np.float32), top_db=45)
        ini = int(idx[0])
        fin = int(idx[1])
        margen = int(sr * 0.15)
        ini = max(0, ini - margen)
        fin = min(len(y), fin + margen)
        if fin > ini + 10:
            return y[ini:fin].astype(np.float32)
    except Exception:
        pass
    abs_y = np.abs(y).astype(np.float32)
    win = max(1, int(sr * 0.02))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env = np.convolve(abs_y, kernel, mode="same")
    pico = float(np.max(env))
    umbral = max(1e-5, 0.005 * pico)
    idx = np.where(env > umbral)[0]
    if len(idx) == 0:
        return y
    ini = int(idx[0])
    fin = int(idx[-1])
    margen = int(sr * 0.10)
    ini = max(0, ini - margen)
    fin = min(len(y) - 1, fin + margen)
    return y[ini : fin + 1]

def envolvente_simple(y, sr, win_seg=0.02):
    if len(y) == 0:
        return y
    abs_y = np.abs(y).astype(np.float32)
    win = max(1, int(sr * float(win_seg)))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(abs_y, kernel, mode="same").astype(np.float32)

def alinear_a_referencia(y_ref, y_rec, sr):
    env_ref = envolvente_simple(y_ref, sr, win_seg=0.02)
    env_rec = envolvente_simple(y_rec, sr, win_seg=0.02)
    paso = max(1, int(sr / 200))
    a = env_rec[::paso]
    b = env_ref[::paso]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    a = a / (float(np.std(a)) + 1e-9)
    b = b / (float(np.std(b)) + 1e-9)
    corr = np.correlate(a, b, mode="full")
    lag_ds = int(np.argmax(corr) - (len(b) - 1))
    lag = int(lag_ds * paso)
    if lag > 0:
        y_rec = y_rec[lag:]
    elif lag < 0:
        y_ref = y_ref[-lag:]
    n = min(len(y_ref), len(y_rec))
    return y_ref[:n], y_rec[:n], lag

def suavizar_en_octavas(freqs_hz, y_db, fmin, fmax, ancho_oct=SUAVIZADO_OCT, puntos=450):
    if len(freqs_hz) < 10:
        raise ValueError("No hay suficientes puntos para suavizar.")
    fmin = float(fmin)
    fmax = float(fmax)
    puntos = int(puntos)
    f_grid = np.logspace(np.log10(fmin), np.log10(fmax), num=puntos).astype(np.float32)
    y_smooth = np.zeros_like(f_grid, dtype=np.float32)
    logf = np.log10(freqs_hz + 1e-9)
    for i, f0 in enumerate(f_grid):
        f1 = float(f0) * (2.0 ** (-float(ancho_oct) / 2.0))
        f2 = float(f0) * (2.0 ** (float(ancho_oct) / 2.0))
        m = (freqs_hz >= f1) & (freqs_hz < f2)
        if np.any(m):
            y_smooth[i] = float(np.median(y_db[m]))
        else:
            y_smooth[i] = float(np.interp(np.log10(float(f0) + 1e-9), logf, y_db))
    return f_grid, y_smooth

def analizar_respuesta_fft(y_ref, y_rec, sr, fmin=FMIN_ANALISIS, fmax=FMAX_ANALISIS):
    nyquist = float(sr) / 2.0
    max_hz = min(float(fmax), nyquist - 1.0)
    min_hz = max(1.0, float(fmin))
    if max_hz <= min_hz:
        raise ValueError("Rango de frecuencias invalido para el analisis.")
    n = min(len(y_ref), len(y_rec))
    if n < 2048:
        raise ValueError("Audio demasiado corto para analizar.")
    y_ref = (y_ref[:n] - float(np.mean(y_ref[:n]))).astype(np.float32)
    y_rec = (y_rec[:n] - float(np.mean(y_rec[:n]))).astype(np.float32)
    w = np.hanning(n).astype(np.float32)
    X = np.fft.rfft(y_ref * w)
    Y = np.fft.rfft(y_rec * w)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr)).astype(np.float32)
    eps = 1e-12
    mag_ref = np.abs(X)
    mag_rec = np.abs(Y)
    db_ref = (20.0 * np.log10(mag_ref + eps)).astype(np.float32)
    db_rec = (20.0 * np.log10(mag_rec + eps)).astype(np.float32)
    resp_db = (db_rec - db_ref).astype(np.float32)
    th = float(np.max(mag_ref)) * 1e-6
    mask = (freqs >= min_hz) & (freqs <= max_hz) & (mag_ref > th)
    if not np.any(mask):
        raise ValueError("No encontre frecuencias utiles. Revisa el archivo de referencia.")
    f = freqs[mask]
    ref = db_ref[mask]
    rec = db_rec[mask]
    resp = resp_db[mask]
    mid_mask = (f >= 200.0) & (f <= 2000.0)
    if np.any(mid_mask):
        offset_resp = float(np.median(resp[mid_mask]))
        ref_mid = float(np.median(ref[mid_mask]))
    else:
        offset_resp = float(np.median(resp))
        ref_mid = float(np.median(ref))
    resp = resp - offset_resp
    ref_rel = ref - ref_mid
    rec_rel = rec - (ref_mid + offset_resp)
    f_plot_min = max(20.0, min_hz)
    f_plot_max = max_hz
    f_grid, resp_s = suavizar_en_octavas(f, resp, f_plot_min, f_plot_max, ancho_oct=SUAVIZADO_OCT)
    _, ref_s = suavizar_en_octavas(f, ref_rel, f_plot_min, f_plot_max, ancho_oct=SUAVIZADO_OCT, puntos=len(f_grid))
    _, rec_s = suavizar_en_octavas(f, rec_rel, f_plot_min, f_plot_max, ancho_oct=SUAVIZADO_OCT, puntos=len(f_grid))
    return {
        "f_raw": f,
        "resp_raw_db": resp,
        "f_grid": f_grid,
        "ref_plot_db": ref_s,
        "rec_plot_db": rec_s,
        "resp_db": resp_s,
    }


TITULO_APP = "AudioLab Quality Comparator"
FONDO = "#101318"
PANEL = "#171c24"
PANEL_SECUNDARIO = "#1e2530"
TEXTO = "#edf2f7"
TEXTO_SECUNDARIO = "#9aa7b6"
ACENTO = "#35c2ff"
ACENTO_2 = "#65d46e"
ADVERTENCIA = "#ffb84d"
PELIGRO = "#ff5d6c"
CUADRICULA = "#303846"


class ComparadorCalidadAudio:
    def __init__(self, root):
        self.root = root
        self.root.title(TITULO_APP)
        self.root.geometry("1320x860")
        self.root.minsize(1120, 720)
        self.root.configure(bg=FONDO)

        self.audio_1 = None
        self.audio_2 = None
        self.analysis = None

        self.configurar_estilo()
        self.construir_interfaz()
        self.establecer_estado("Listo. Carga el audio calibrado y el audio grabado para iniciar.")

    def configurar_estilo(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=FONDO)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=FONDO, foreground=TEXTO, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=TEXTO_SECUNDARIO, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=FONDO, foreground=TEXTO, font=("Segoe UI Semibold", 22))
        style.configure("Metric.TLabel", background=PANEL_SECUNDARIO, foreground=TEXTO, font=("Consolas", 10))
        style.configure("Status.TLabel", background="#0b0e13", foreground=TEXTO_SECUNDARIO, font=("Segoe UI", 9))
        style.configure("TNotebook", background=FONDO, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=TEXTO_SECUNDARIO, padding=(16, 8), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL_SECUNDARIO)], foreground=[("selected", TEXTO)])

        style.configure(
            "Accent.TButton",
            background=ACENTO,
            foreground="#071016",
            borderwidth=0,
            focusthickness=0,
            padding=(16, 10),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#6fd6ff"), ("disabled", "#303846")])
        style.configure(
            "Soft.TButton",
            background=PANEL_SECUNDARIO,
            foreground=TEXTO,
            borderwidth=0,
            focusthickness=0,
            padding=(14, 9),
            font=("Segoe UI", 10),
        )
        style.map("Soft.TButton", background=[("active", "#2a3342"), ("disabled", "#232936")])

    def construir_interfaz(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(22, 18, 22, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=TITULO_APP, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Comparacion visual y estadistica para calidad de bocinas usando audio calibrado vs audio grabado",
            foreground=TEXTO_SECUNDARIO,
            background=FONDO,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        main = ttk.Frame(self.root, padding=(22, 8, 22, 0))
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0, minsize=330)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.construir_panel_lateral(main)
        self.construir_area_graficas(main)

        self.status_var = tk.StringVar()
        status = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel", padding=(16, 7))
        status.grid(row=2, column=0, sticky="ew")

    def construir_panel_lateral(self, parent):
        sidebar = ttk.Frame(parent, style="Panel.TFrame", padding=16)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 16))
        sidebar.columnconfigure(0, weight=1)

        ttk.Label(sidebar, text="Archivos WAV", background=PANEL, foreground=TEXTO, font=("Segoe UI Semibold", 13)).grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )

        ttk.Button(sidebar, text="Cargar Audio 1 calibrado", style="Accent.TButton", command=lambda: self.cargar_audio(1)).grid(
            row=1, column=0, sticky="ew", pady=(0, 8)
        )
        self.audio_1_name = tk.StringVar(value="Sin archivo")
        self.audio_1_info = tk.StringVar(value="Duracion: -- | Fs: --")
        ttk.Label(sidebar, textvariable=self.audio_1_name, style="Muted.TLabel", wraplength=285).grid(row=2, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self.audio_1_info, style="Muted.TLabel").grid(row=3, column=0, sticky="w", pady=(2, 16))

        ttk.Button(sidebar, text="Cargar Audio 2 grabado", style="Accent.TButton", command=lambda: self.cargar_audio(2)).grid(
            row=4, column=0, sticky="ew", pady=(0, 8)
        )
        self.audio_2_name = tk.StringVar(value="Sin archivo")
        self.audio_2_info = tk.StringVar(value="Duracion: -- | Fs: --")
        ttk.Label(sidebar, textvariable=self.audio_2_name, style="Muted.TLabel", wraplength=285).grid(row=5, column=0, sticky="w")
        ttk.Label(sidebar, textvariable=self.audio_2_info, style="Muted.TLabel").grid(row=6, column=0, sticky="w", pady=(2, 18))

        ttk.Separator(sidebar).grid(row=7, column=0, sticky="ew", pady=(0, 16))

        ttk.Button(sidebar, text="Comparar Audios", style="Accent.TButton", command=self.comparar_audios).grid(
            row=8, column=0, sticky="ew", pady=(0, 8)
        )
        ttk.Button(sidebar, text="Limpiar", style="Soft.TButton", command=self.limpiar_todo).grid(
            row=9, column=0, sticky="ew", pady=(0, 18)
        )

        ttk.Label(sidebar, text="Resultado", background=PANEL, foreground=TEXTO, font=("Segoe UI Semibold", 13)).grid(
            row=10, column=0, sticky="w"
        )
        self.result_var = tk.StringVar(value="Esperando comparacion.")
        result = tk.Label(
            sidebar,
            textvariable=self.result_var,
            bg=PANEL_SECUNDARIO,
            fg=TEXTO,
            font=("Segoe UI", 11),
            justify="left",
            anchor="nw",
            wraplength=285,
            padx=12,
            pady=12,
        )
        result.grid(row=11, column=0, sticky="nsew", pady=(8, 0))
        sidebar.rowconfigure(11, weight=1)

    def construir_area_graficas(self, parent):
        graph_panel = ttk.Frame(parent, style="Panel.TFrame", padding=10)
        graph_panel.grid(row=0, column=1, sticky="nsew")
        graph_panel.columnconfigure(0, weight=1)
        graph_panel.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(graph_panel)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.figures = {}
        self.canvases = {}
        tabs = [
            ("fft", "Espectros (dB)"),
            ("comparison", "Respuesta en Frecuencia"),
        ]

        for key, label in tabs:
            frame = ttk.Frame(self.notebook, style="Panel.TFrame")
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            self.notebook.add(frame, text=label)

            figure = Figure(figsize=(9, 6), dpi=100, facecolor=PANEL)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            toolbar_frame = ttk.Frame(frame, style="Panel.TFrame")
            toolbar_frame.grid(row=1, column=0, sticky="ew")
            NavigationToolbar2Tk(canvas, toolbar_frame).update()

            self.figures[key] = figure
            self.canvases[key] = canvas

        self.dibujar_graficas_vacias()

    def establecer_estado(self, mensaje):
        self.status_var.set(mensaje)
        self.root.update_idletasks()

    def cargar_audio(self, slot):
        path = filedialog.askopenfilename(
            title="Selecciona un archivo WAV",
            filetypes=[("Archivos WAV", "*.wav")],
        )
        if not path:
            return

        try:
            if not os.path.exists(path):
                raise FileNotFoundError("El archivo seleccionado no existe.")
            if not path.lower().endswith(".wav"):
                raise ValueError("Solo se permiten archivos .wav.")

            self.establecer_estado("Cargando audio " + str(slot) + "...")
            data, sample_rate = sf.read(path, always_2d=False)
            if data.size == 0:
                raise ValueError("El archivo WAV no contiene muestras.")

            data = self.convertir_a_mono(data)
            data = recortar_silencios(data, sample_rate)
            data = self.normalizar_audio(data)
            duration = len(data) / sample_rate
            stats = self.calcular_estadisticas_audio(data, sample_rate)

            audio = {
                "path": path,
                "name": os.path.basename(path),
                "data": data,
                "sample_rate": sample_rate,
                "duration": duration,
                "stats": stats,
                "fft": self.calcular_fft(data, sample_rate),
            }

            if slot == 1:
                self.audio_1 = audio
                self.audio_1_name.set(audio["name"])
                self.audio_1_info.set(self.informacion_audio_lateral(audio))
            else:
                self.audio_2 = audio
                self.audio_2_name.set(audio["name"])
                self.audio_2_info.set(self.informacion_audio_lateral(audio))

            self.analysis = None
            self.graficar_fft()
            self.establecer_estado("Audio " + str(slot) + " cargado correctamente.")
        except Exception as error:
            messagebox.showerror("Error al cargar audio", str(error))
            self.establecer_estado("No se pudo cargar el audio.")

    def convertir_a_mono(self, data):
        data = np.asarray(data, dtype=np.float64)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data

    def normalizar_audio(self, data):
        peak = np.max(np.abs(data))
        if peak == 0:
            return data
        return data / peak

    def calcular_fft(self, data, sample_rate):
        if len(data) < 2:
            return {"freqs": np.array([]), "magnitudes": np.array([]), "peaks": np.array([])}

        window = np.hanning(len(data))
        spectrum = np.abs(rfft(data * window))
        freqs = rfftfreq(len(data), 1 / sample_rate)
        magnitudes = spectrum / max(len(data), 1)
        if len(magnitudes) > 0 and np.max(magnitudes) > 0:
            magnitudes = magnitudes / np.max(magnitudes)

        threshold = 0.12 if len(magnitudes) < 5000 else 0.08
        peaks, _ = find_peaks(magnitudes, height=threshold, distance=max(5, len(magnitudes) // 700))
        peaks = peaks[np.argsort(magnitudes[peaks])[-8:]] if len(peaks) else peaks
        return {"freqs": freqs, "magnitudes": magnitudes, "peaks": peaks}

    def calcular_estadisticas_audio(self, data, sample_rate):
        fft = self.calcular_fft(data, sample_rate)
        magnitudes = fft["magnitudes"]
        freqs = fft["freqs"]
        dominant = 0
        if len(magnitudes) > 1:
            dominant = freqs[int(np.argmax(magnitudes[1:]) + 1)]

        rms = float(np.sqrt(np.mean(np.square(data)))) if len(data) else 0
        energy = float(np.sum(np.square(data))) if len(data) else 0
        return {
            "duration": len(data) / sample_rate if sample_rate else 0,
            "sample_rate": sample_rate,
            "rms": rms,
            "energy": energy,
            "dominant_frequency": float(dominant),
        }

    def comparar_audios(self):
        if self.audio_1 is None or self.audio_2 is None:
            messagebox.showwarning("Faltan audios", "Carga el audio calibrado y el audio grabado antes de comparar.")
            return

        try:
            self.establecer_estado("Analizando correlacion, FFT y similitud espectral...")
            self.root.after(50, self.ejecutar_comparacion)
        except Exception as error:
            messagebox.showerror("Error de comparacion", str(error))
            self.establecer_estado("No se pudo comparar los audios.")

    def ejecutar_comparacion(self):
        try:
            a1 = self.audio_1["data"]
            a2 = self.audio_2["data"]
            sr = min(self.audio_1["sample_rate"], self.audio_2["sample_rate"])

            if self.audio_1["sample_rate"] != sr:
                a1 = librosa.resample(a1, orig_sr=self.audio_1["sample_rate"], target_sr=sr)
            if self.audio_2["sample_rate"] != sr:
                a2 = librosa.resample(a2, orig_sr=self.audio_2["sample_rate"], target_sr=sr)

            min_len = min(len(a1), len(a2))
            if min_len < 4:
                raise ValueError("Los audios son demasiado cortos para compararse.")

            a1, a2, lag = alinear_a_referencia(a1, a2, sr)
            res = analizar_respuesta_fft(a1, a2, sr)

            std_resp = float(np.std(res["resp_db"]))
            similarity = max(0, 100 - std_resp * 10)
            label = "Excelente" if similarity >= 85 else "Aceptable" if similarity >= 60 else "Pobre"

            self.analysis = {
                "sample_rate": sr,
                "similarity": similarity,
                "label": label,
                "aligned_1": a1,
                "aligned_2": a2,
                "res": res,
                "std_resp": std_resp,
            }

            self.result_var.set(
                "Similitud de curva: " + self.formatear_porcentaje(similarity) + "\n"
                "Calidad: " + label + "\n\n"
                "Desviacion estandar (dB): " + self.formatear_numero(std_resp, 2) + "\n"
                "Rango analiz.: " + str(int(FMIN_ANALISIS)) + " - " + str(int(FMAX_ANALISIS)) + " Hz"
            )

            self.graficar_comparacion()
            self.graficar_fft()
            self.establecer_estado("Analisis completado: " + label + " (" + self.formatear_porcentaje(similarity) + ").")
        except Exception as error:
            messagebox.showerror("Error de analisis", str(error))
            self.establecer_estado("No se pudo analizar los audios.")

    def dibujar_graficas_vacias(self):
        for key, figure in self.figures.items():
            figure.clear()
            axis = figure.add_subplot(111)
            self.aplicar_estilo_eje(axis)
            axis.text(0.5, 0.5, "Carga archivos WAV para visualizar el analisis", ha="center", va="center", color=TEXTO_SECUNDARIO)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.tight_layout()
            self.canvases[key].draw()


    def graficar_fft(self):
        figure = self.figures["fft"]
        figure.clear()
        if self.analysis is None:
            axis = figure.add_subplot(111)
            self.aplicar_estilo_eje(axis)
            axis.text(0.5, 0.5, "Ejecuta la comparacion para ver espectros", ha="center", va="center", color=TEXTO_SECUNDARIO)
            self.canvases["fft"].draw()
            return
            
        axes = figure.subplots(2, 1, sharex=True)
        res = self.analysis["res"]
        f = res["f_grid"]
        refp = res["ref_plot_db"]
        recp = res["rec_plot_db"]
        
        ticks_x = [20, 40, 80, 160, 320, 640, 1250, 2500, 5000, 10000, 20000]
        labels_x = [f"{t} Hz" if t < 1000 else f"{t/1000:g} kHz" for t in ticks_x]

        self.aplicar_estilo_eje(axes[0])
        axes[0].semilogx(f, refp, color=ACENTO)
        axes[0].set_xlim(float(f[0]), float(f[-1]))
        axes[0].set_ylabel("Nivel (dB rel)", color=TEXTO_SECUNDARIO)
        axes[0].set_title("Señal de Referencia", color=TEXTO, fontsize=12, loc="left")
        
        self.aplicar_estilo_eje(axes[1])
        axes[1].semilogx(f, recp, color=ACENTO_2)
        axes[1].set_xlim(float(f[0]), float(f[-1]))
        axes[1].set_xticks(ticks_x)
        axes[1].set_xticklabels(labels_x)
        axes[1].set_xlabel("Frecuencia", color=TEXTO_SECUNDARIO)
        axes[1].set_ylabel("Nivel (dB rel)", color=TEXTO_SECUNDARIO)
        axes[1].set_title("Señal Grabada", color=TEXTO, fontsize=12, loc="left")

        figure.tight_layout(pad=2.0)
        self.canvases["fft"].draw()

    def graficar_comparacion(self):
        figure = self.figures["comparison"]
        figure.clear()
        if self.analysis is None:
            axis = figure.add_subplot(111)
            self.aplicar_estilo_eje(axis)
            axis.text(0.5, 0.5, "Ejecuta la comparacion para ver resultados", ha="center", va="center", color=TEXTO_SECUNDARIO)
            self.canvases["comparison"].draw()
            return

        axis = figure.add_subplot(111)
        self.aplicar_estilo_eje(axis)
        
        res = self.analysis["res"]
        f = res["f_grid"]
        resp = res["resp_db"]
        
        ticks_x = [20, 40, 80, 160, 320, 640, 1250, 2500, 5000, 10000, 20000]
        labels_x = [f"{t} Hz" if t < 1000 else f"{t/1000:g} kHz" for t in ticks_x]

        axis.semilogx(res["f_raw"], res["resp_raw_db"], color=ACENTO, alpha=0.15, label="Cruda")
        axis.semilogx(f, resp, color=ACENTO, linewidth=2.5, label="Suavizada (Grabación - Referencia)")
        axis.axhline(0.0, color=TEXTO, linewidth=1, alpha=0.4, label="Referencia Ideal (0 dB)")
        axis.axhline(3.0, color=TEXTO, linewidth=1, alpha=0.15, linestyle="--")
        axis.axhline(-3.0, color=TEXTO, linewidth=1, alpha=0.15, linestyle="--")
        axis.axhspan(-3.0, 3.0, color=TEXTO, alpha=0.06)

        axis.set_xlim(float(f[0]), float(f[-1]))
        axis.set_xticks(ticks_x)
        axis.set_xticklabels(labels_x)
        axis.set_xlabel("Frecuencia", color=TEXTO_SECUNDARIO)
        axis.set_ylabel("Diferencia de Volumen (dB)", color=TEXTO_SECUNDARIO)
        axis.set_title("Respuesta de Frecuencia (Comparación: Grabación vs Referencia)", color=TEXTO, fontsize=13, loc="left")
        axis.legend(facecolor=PANEL_SECUNDARIO, edgecolor=CUADRICULA, labelcolor=TEXTO, loc="lower right")

        y_min = float(np.min(resp))
        y_max = float(np.max(resp))
        y_min = max(-40.0, y_min - 5.0)
        y_max = min(40.0, y_max + 5.0)
        axis.set_ylim(y_min, y_max)

        figure.tight_layout(pad=2.0)
        self.canvases["comparison"].draw()
        self.notebook.select(1)

    def aplicar_estilo_eje(self, axis):
        axis.set_facecolor(PANEL)
        axis.tick_params(colors=TEXTO_SECUNDARIO)
        axis.grid(True, color=CUADRICULA, alpha=0.55, linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color(CUADRICULA)
        axis.xaxis.label.set_color(TEXTO_SECUNDARIO)
        axis.yaxis.label.set_color(TEXTO_SECUNDARIO)

    def limpiar_todo(self):
        self.audio_1 = None
        self.audio_2 = None
        self.analysis = None
        self.audio_1_name.set("Sin archivo")
        self.audio_2_name.set("Sin archivo")
        self.audio_1_info.set("Duracion: -- | Fs: --")
        self.audio_2_info.set("Duracion: -- | Fs: --")
        self.result_var.set("Esperando comparacion.")
        self.dibujar_graficas_vacias()
        self.establecer_estado("Datos limpiados. Carga nuevos archivos WAV.")

    def formatear_segundos(self, seconds):
        return self.formatear_numero(seconds, 2) + " s"

    def informacion_audio_lateral(self, audio):
        stats = audio["stats"]
        return (
            "Duracion: " + self.formatear_segundos(stats["duration"]) + "\n"
            "Fs: " + str(stats["sample_rate"]) + " Hz | RMS: " + self.formatear_numero(stats["rms"], 5)
        )

    def formatear_porcentaje(self, value):
        return self.formatear_numero(value, 2) + "%"

    def formatear_numero(self, value, decimals):
        return ("{:." + str(decimals) + "f}").format(float(value))


def main():
    root = tk.Tk()
    app = ComparadorCalidadAudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()








