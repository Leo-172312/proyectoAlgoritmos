import os
import numpy as np

FMIN_ANALISIS = 20.0
FMAX_ANALISIS = 20000.0
SUAVIZADO_OCT = 1.0 / 3.0


def cargar_audio(ruta):
    if not os.path.isfile(ruta):
        raise FileNotFoundError("No se encontro el archivo: " + ruta)

    ultimo_error = None

    try:
        import librosa  # type: ignore
        y, sr = librosa.load(ruta, sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except ImportError:
        pass
    except Exception as e:
        ultimo_error = e

    try:
        from scipy.io import wavfile  # type: ignore
        sr, data = wavfile.read(ruta)

        if hasattr(data, "ndim") and data.ndim == 2:
            data = data.mean(axis=1)

        if data.dtype == np.int16:
            y = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            y = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            y = (data.astype(np.float32) - 128.0) / 128.0
        else:
            y = data.astype(np.float32)

        return y, int(sr)
    except Exception:
        if ultimo_error is not None:
            raise ValueError(
                "No pude leer el audio. Prueba con WAV."
                "Error original: " + str(ultimo_error)
            )
        raise ValueError("No pude leer el audio. Prueba con un archivo WAV")


def recortar_silencios(y, sr):
    if len(y) == 0:
        return y

    try:
        import librosa  # type: ignore
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


def remuestrear(y, sr_origen, sr_destino):
    if sr_origen == sr_destino:
        return y

    try:
        from scipy.signal import resample_poly  # type: ignore
        from math import gcd

        g = gcd(sr_origen, sr_destino)
        up = sr_destino // g
        down = sr_origen // g
        return resample_poly(y, up, down).astype(np.float32)
    except Exception:
        dur = len(y) / float(sr_origen)
        nuevo_n = max(1, int(dur * sr_destino))
        t_viejo = np.linspace(0.0, dur, num=len(y), endpoint=False)
        t_nuevo = np.linspace(0.0, dur, num=nuevo_n, endpoint=False)
        return np.interp(t_nuevo, t_viejo, y).astype(np.float32)


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


def graficas_sweep(r):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("Matplotlib no esta instalado, no se puede mostrar grafica.")
        return

    f = r["f_grid"]
    refp = r["ref_plot_db"]
    recp = r["rec_plot_db"]
    resp = r["resp_db"]

    ticks_x = [20, 40, 80, 160, 320, 640, 1250, 2500, 5000, 10000, 20000]
    labels_x = [f"{t} Hz" if t < 1000 else f"{t/1000:g} kHz" for t in ticks_x]

    plt.figure(figsize=(10, 4))
    plt.semilogx(f, refp)
    plt.xlim(float(f[0]), float(f[-1]))
    plt.xticks(ticks_x, labels_x)
    plt.xlabel("Frecuencia")
    plt.ylabel("Nivel (dB relativo)")
    plt.title("Señal de Referencia: Audio original enviado a las bocinas")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.semilogx(f, recp)
    plt.xlim(float(f[0]), float(f[-1]))
    plt.xticks(ticks_x, labels_x)
    plt.xlabel("Frecuencia")
    plt.ylabel("Nivel (dB relativo)")
    plt.title("Señal Grabada: Audio capturado por el micrófono")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.semilogx(r["f_raw"], r["resp_raw_db"], alpha=0.15)
    plt.semilogx(f, resp, linewidth=2.5)
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.axhline(3.0, color="black", linewidth=1, alpha=0.15, linestyle="--")
    plt.axhline(-3.0, color="black", linewidth=1, alpha=0.15, linestyle="--")
    plt.axhspan(-3.0, 3.0, alpha=0.06)

    plt.xlim(float(f[0]), float(f[-1]))
    plt.xticks(ticks_x, labels_x)
    plt.xlabel("Frecuencia")
    plt.ylabel("Diferencia de Volumen (dB)")
    plt.title("Respuesta de Frecuencia: Qué tan planas suenan las bocinas (0 dB = perfecto)")
    plt.grid(True, which="both", alpha=0.25)

    y_min = float(np.min(resp))
    y_max = float(np.max(resp))
    y_min = max(-40.0, y_min - 5.0)
    y_max = min(40.0, y_max + 5.0)
    plt.ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()


def main():
    print("Analizador de calidad de bocinas")
    print("")

    ruta_ref = input("Ruta del audio de referencia: ").strip().strip('"')
    ruta_rec_txt = input(
        "Ruta del audio grabado: "
    ).strip()
    rutas_rec = [p.strip().strip('"') for p in ruta_rec_txt.split(",") if p.strip()]

    if len(rutas_rec) == 0:
        print("Error: No escribiste ninguna ruta de grabacion.")
        return

    try:
        y_ref, sr_ref = cargar_audio(ruta_ref)
        y_ref = recortar_silencios(y_ref, sr_ref)

        y_rec, sr_rec = cargar_audio(rutas_rec[0])
        y_rec = recortar_silencios(y_rec, sr_rec)

        sr = min(sr_ref, sr_rec)
        if sr_ref != sr:
            y_ref = remuestrear(y_ref, sr_ref, sr)
        if sr_rec != sr:
            y_rec = remuestrear(y_rec, sr_rec, sr)

        y_ref, y_rec, _ = alinear_a_referencia(y_ref, y_rec, sr)
        res = analizar_respuesta_fft(y_ref, y_rec, sr)

        graficas_sweep(res)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()