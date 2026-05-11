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
            ("wave", "Formas de onda"),
            ("fft", "FFT"),
            ("comparison", "Comparacion"),
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
            self.graficar_formas_onda()
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

            a1 = a1[:min_len]
            a2 = a2[:min_len]
            a1 = a1 - np.mean(a1)
            a2 = a2 - np.mean(a2)

            corr = float(np.corrcoef(a1, a2)[0, 1])
            if np.isnan(corr):
                corr = 0
            corr_score = max(0, corr) * 100

            fft1 = self.calcular_fft(a1, sr)
            fft2 = self.calcular_fft(a2, sr)
            fft_len = min(len(fft1["magnitudes"]), len(fft2["magnitudes"]))
            spectral_distance = float(np.mean(np.abs(fft1["magnitudes"][:fft_len] - fft2["magnitudes"][:fft_len])))
            spectral_score = max(0, 100 * (1 - spectral_distance))

            rms1 = self.audio_1["stats"]["rms"]
            rms2 = self.audio_2["stats"]["rms"]
            rms_score = 100 * (1 - min(abs(rms1 - rms2) / max(rms1, rms2, 0.000001), 1))

            dom1 = self.audio_1["stats"]["dominant_frequency"]
            dom2 = self.audio_2["stats"]["dominant_frequency"]
            dominant_score = 100 * (1 - min(abs(dom1 - dom2) / max(dom1, dom2, 1), 1))

            similarity = (corr_score * 0.35) + (spectral_score * 0.45) + (rms_score * 0.10) + (dominant_score * 0.10)
            label = "Muy similares" if similarity >= 82 else "Similares" if similarity >= 58 else "Diferentes"

            self.analysis = {
                "sample_rate": sr,
                "correlation": corr,
                "correlation_score": corr_score,
                "spectral_score": spectral_score,
                "rms_score": rms_score,
                "dominant_score": dominant_score,
                "similarity": similarity,
                "label": label,
                "spectral_distance": spectral_distance,
                "aligned_1": a1,
                "aligned_2": a2,
                "fft_1": fft1,
                "fft_2": fft2,
            }

            self.result_var.set(
                "Similitud: " + self.formatear_porcentaje(similarity) + "\n"
                "Resultado: " + label + "\n\n"
                "Correlacion: " + self.formatear_numero(corr, 4) + "\n"
                "Score espectral: " + self.formatear_porcentaje(spectral_score) + "\n"
                "RMS calibrado: " + self.formatear_numero(rms1, 5) + "\n"
                "RMS grabado: " + self.formatear_numero(rms2, 5) + "\n"
                "Freq. dominante A1: " + self.formatear_numero(dom1, 1) + " Hz\n"
                "Freq. dominante A2: " + self.formatear_numero(dom2, 1) + " Hz"
            )
            self.graficar_comparacion()
            self.graficar_fft()
            self.establecer_estado("Comparacion completada: " + label + " (" + self.formatear_porcentaje(similarity) + ").")
        except Exception as error:
            messagebox.showerror("Error de comparacion", str(error))
            self.establecer_estado("No se pudo comparar los audios.")

    def dibujar_graficas_vacias(self):
        for key, figure in self.figures.items():
            figure.clear()
            axis = figure.add_subplot(111)
            self.aplicar_estilo_eje(axis)
            axis.TEXTO(0.5, 0.5, "Carga archivos WAV para visualizar el analisis", ha="center", va="center", color=TEXTO_SECUNDARIO)
            axis.set_xticks([])
            axis.set_yticks([])
            figure.tight_layout()
            self.canvases[key].draw()

    def graficar_formas_onda(self):
        figure = self.figures["wave"]
        figure.clear()
        axes = figure.subplots(2, 1, sharex=False)
        self.graficar_forma_onda_individual(axes[0], self.audio_1, "Audio 1 calibrado", ACENTO)
        self.graficar_forma_onda_individual(axes[1], self.audio_2, "Audio 2 grabado", ACENTO_2)
        figure.tight_layout(pad=2.0)
        self.canvases["wave"].draw()

    def graficar_forma_onda_individual(self, axis, audio, title, color):
        self.aplicar_estilo_eje(axis)
        axis.set_title(title, color=TEXTO, fontsize=12, loc="left")
        if audio is None:
            axis.TEXTO(0.5, 0.5, "Sin audio cargado", ha="center", va="center", color=TEXTO_SECUNDARIO, transform=axis.transAxes)
            return
        data = audio["data"]
        time = np.linspace(0, audio["duration"], len(data))
        step = max(1, len(data) // 25000)
        axis.plot(time[::step], data[::step], color=color, linewidth=0.8)
        axis.set_ylabel("Amplitud", color=TEXTO_SECUNDARIO)
        axis.set_xlabel("Tiempo (s)", color=TEXTO_SECUNDARIO)
        axis.set_ylim(-1.05, 1.05)
    def graficar_fft(self):
        figure = self.figures["fft"]
        figure.clear()
        axis = figure.add_subplot(111)
        self.aplicar_estilo_eje(axis)
        axis.set_title("Espectro de frecuencias FFT", color=TEXTO, fontsize=13, loc="left")
        axis.set_xlabel("Frecuencia (Hz)", color=TEXTO_SECUNDARIO)
        axis.set_ylabel("Magnitud normalizada", color=TEXTO_SECUNDARIO)

        self.graficar_linea_fft(axis, self.audio_1, "Audio 1 calibrado", ACENTO)
        self.graficar_linea_fft(axis, self.audio_2, "Audio 2 grabado", ACENTO_2)

        if self.analysis is not None:
            fft1 = self.analysis["fft_1"]
            fft2 = self.analysis["fft_2"]
            length = min(len(fft1["freqs"]), len(fft2["freqs"]))
            if length:
                diff = np.abs(fft1["magnitudes"][:length] - fft2["magnitudes"][:length])
                notable = np.where(diff > np.percentile(diff, 98))[0]
                if len(notable):
                    axis.scatter(fft1["freqs"][notable], diff[notable], s=8, color=ADVERTENCIA, alpha=0.45, label="Diferencias relevantes")

        axis.set_xlim(0, self.obtener_limite_fft())
        axis.legend(facecolor=PANEL_SECUNDARIO, edgecolor=CUADRICULA, labelcolor=TEXTO)
        figure.tight_layout()
        self.canvases["fft"].draw()

    def graficar_linea_fft(self, axis, audio, label, color):
        if audio is None:
            return
        fft = audio["fft"]
        freqs = fft["freqs"]
        magnitudes = fft["magnitudes"]
        if len(freqs) == 0:
            return
        limit = self.obtener_limite_fft()
        visible = freqs <= limit
        axis.plot(freqs[visible], magnitudes[visible], color=color, linewidth=1.0, alpha=0.9, label=label)

        peaks = fft["peaks"]
        if len(peaks):
            peaks = peaks[freqs[peaks] <= limit]
            axis.scatter(freqs[peaks], magnitudes[peaks], color=color, s=32, edgecolors=TEXTO, linewidths=0.4)
            for peak in peaks[np.argsort(magnitudes[peaks])[-3:]]:
                axis.annotate(
                    self.formatear_numero(freqs[peak], 0) + " Hz",
                    (freqs[peak], magnitudes[peak]),
                    textcoords="offset points",
                    xytext=(4, 7),
                    color=TEXTO,
                    fontsize=8,
                )

    def obtener_limite_fft(self):
        rates = []
        if self.audio_1 is not None:
            rates.append(self.audio_1["sample_rate"])
        if self.audio_2 is not None:
            rates.append(self.audio_2["sample_rate"])
        if not rates:
            return 12000
        return min(20000, max(rates) / 2)

    def graficar_comparacion(self):
        figure = self.figures["comparison"]
        figure.clear()
        if self.analysis is None:
            axis = figure.add_subplot(111)
            self.aplicar_estilo_eje(axis)
            axis.TEXTO(0.5, 0.5, "Ejecuta la comparacion para ver resultados", ha="center", va="center", color=TEXTO_SECUNDARIO)
            self.canvases["comparison"].draw()
            return

        CUADRICULA = figure.add_gridspec(2, 2)
        bars = figure.add_subplot(CUADRICULA[0, 0])
        diff_axis = figure.add_subplot(CUADRICULA[0, 1])
        overlay = figure.add_subplot(CUADRICULA[1, :])

        for axis in [bars, diff_axis, overlay]:
            self.aplicar_estilo_eje(axis)

        labels = ["Correlacion", "Espectral", "RMS", "Dominante", "Total"]
        values = [
            self.analysis["correlation_score"],
            self.analysis["spectral_score"],
            self.analysis["rms_score"],
            self.analysis["dominant_score"],
            self.analysis["similarity"],
        ]
        colors = [ACENTO, ACENTO, ACENTO_2, ACENTO_2, ADVERTENCIA]
        bars.bar(labels, values, color=colors)
        bars.set_ylim(0, 100)
        bars.set_title("Indicadores de similitud", color=TEXTO, fontsize=12, loc="left")
        bars.tick_params(axis="x", labelrotation=20)

        fft1 = self.analysis["fft_1"]
        fft2 = self.analysis["fft_2"]
        length = min(len(fft1["freqs"]), len(fft2["freqs"]))
        if length:
            freqs = fft1["freqs"][:length]
            diff = np.abs(fft1["magnitudes"][:length] - fft2["magnitudes"][:length])
            visible = freqs <= self.obtener_limite_fft()
            diff_axis.fill_between(freqs[visible], diff[visible], color=PELIGRO, alpha=0.55)
            diff_axis.set_xlim(0, self.obtener_limite_fft())
        diff_axis.set_title("Diferencia espectral", color=TEXTO, fontsize=12, loc="left")
        diff_axis.set_xlabel("Frecuencia (Hz)", color=TEXTO_SECUNDARIO)

        sr = self.analysis["sample_rate"]
        time = np.linspace(0, len(self.analysis["aligned_1"]) / sr, len(self.analysis["aligned_1"]))
        step = max(1, len(time) // 22000)
        overlay.plot(time[::step], self.analysis["aligned_1"][::step], color=ACENTO, linewidth=0.8, label="Calibrado")
        overlay.plot(time[::step], self.analysis["aligned_2"][::step], color=ACENTO_2, linewidth=0.8, alpha=0.75, label="Grabado")
        overlay.set_title("Ondas alineadas para comparacion", color=TEXTO, fontsize=12, loc="left")
        overlay.set_xlabel("Tiempo (s)", color=TEXTO_SECUNDARIO)
        overlay.legend(facecolor=PANEL_SECUNDARIO, edgecolor=CUADRICULA, labelcolor=TEXTO)

        figure.suptitle(
            self.analysis["label"] + " - similitud " + self.formatear_porcentaje(self.analysis["similarity"]),
            color=TEXTO,
            fontsize=14,
        )
        figure.tight_layout(pad=2.0)
        self.canvases["comparison"].draw()
        self.notebook.select(2)

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






