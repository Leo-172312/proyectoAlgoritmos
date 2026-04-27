import numpy as np
import matplotlib.pyplot as plt

#este codigo demuestra cómo una señal en el tiempo se puede descomponer en sus frecuencias usando la FFT. 

# Parámetros de la señal
fs = 1000  # frecuencia de muestreo (Hz), 1000 muestras por segundo
t = np.linspace(0, 1, fs, endpoint=False)  # 1 segundo


# Crear señal (mezcla de frecuencias)
f1 = 50   # Hz
f2 = 120  # Hz
f3 = 300  # Hz

# Crear señal que la suma de tres frecuencias diferentes
senal = (np.sin(2 * np.pi * f1 * t) +
         0.5 * np.sin(2 * np.pi * f2 * t) +
         0.2 * np.sin(2 * np.pi * f3 * t))


# Aplicar FFT
fft_senal = np.fft.fft(senal) #La FFT descompone la señal en todas las frecuencias que la componen.
frecuencias = np.fft.fftfreq(len(fft_senal), 1/fs) #Se obtiene el eje de frecuencias para interpretar el resultado


# Tomar solo parte positiva
mitad = len(fft_senal) // 2 #Se queda con la mitad positiva la otra la descarta
fft_magnitud = np.abs(fft_senal[:mitad]) #Nos indica que tan fuerte esta cada frecuencia
frecuencias = frecuencias[:mitad]


# Graficar
plt.figure(figsize=(12,5))


# Señal en el tiempo no se distingue las frecuencias que contiene
plt.subplot(1,2,1)
plt.plot(t, senal)
plt.title("Señal en el tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")


# Espectro de frecuencias
plt.subplot(1,2,2)
plt.plot(frecuencias, fft_magnitud)
plt.title("Espectro de frecuencias (FFT)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")


plt.tight_layout()
plt.show()