"""
Práctica 2: Sistema cardiovascular

Departamento de Ingeniería Eléctrica y Electrónica, Ingeniería Biomédica
Tecnológico Nacional de México [TecNM - Tijuana]
Blvd. Alberto Limón Padilla s/n, C.P. 22454, Tijuana, B.C., México

Nombre del alumno: Osiris Jaylin Chavez Hernandez
Número de control: 23210697
Correo institucional: l23210697@tectijuana.edu.mx

Asignatura: Modelado de Sistemas Fisiológicos
Docente: Dr. Paul Antonio Valle Trujillo; paul.valle@tectijuana.edu.mx
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from scipy import signal
import pandas as pd

u = np.array(pd.read_excel('signal.xlsx', header=None))

x0 = 0
t0 = 0
tend = 15
dt = 1e-3
w, h = 7, 4.5

N = round((tend - t0) / dt) + 1
t = np.linspace(t0, tend, N)

u = np.reshape(signal.resample(u, len(t)), -1)

def cardio(Z, C, R, L):
    num = [L * R, R * Z]
    den = [C * L * R * Z, L * R + L * Z, R * Z]
    sys = ctrl.tf(num, den)
    return sys

Z, C, R, L = 0.033, 1.5, 0.95, 0.01
sysnormo = cardio(Z, C, R, L)
print("Función de transferencia del normotenso:")
print(sysnormo)

Z, C, R, L = 0.02, 0.25, 0.6, 0.005
syshipo = cardio(Z, C, R, L)
print("Función de transferencia del hipotenso:")
print(syshipo)

Z, C, R, L = 0.05, 2.5, 1.4, 0.02
syshiper = cardio(Z, C, R, L)
print("Función de transferencia del hipertenso:")
print(syshiper)

_, Pp1 = ctrl.forced_response(sysnormo, T=t, U=u, X0=x0)
_, Pp2 = ctrl.forced_response(syshipo, T=t, U=u, X0=x0)
_, Pp3 = ctrl.forced_response(syshiper, T=t, U=u, X0=x0)

fig1, ax = plt.subplots(figsize=(8, 4.8), facecolor='white')

ax.plot(t, Pp1, '-', linewidth=1.2, label='Pp(t): Normotenso')
ax.plot(t, Pp2, '-', linewidth=1.2, label='Pp(t): Hipotenso')
ax.plot(t, Pp3, '-', linewidth=1.2, label='Pp(t): Hipertenso')

y_all = np.concatenate([Pp1, Pp2, Pp3])
ymin, ymax = np.min(y_all), np.max(y_all)
ymargin = 0.08 * (ymax - ymin if ymax != ymin else 1)

ax.set_xlim(t[0], t[-1])
ax.set_ylim(ymin - ymargin, ymax + ymargin)
ax.set_xticks(np.arange(0, 16, 1))

ax.margins(x=0.01)

ax.set_xlabel('t [s]')
ax.set_ylabel('Vi(t) [V]')
ax.set_title('Respuestas en lazo abierto')
ax.grid(True, alpha=0.25)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

fig1.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.25)
plt.savefig('Cardiovascular_lazo_abierto_python.pdf', format='pdf')
plt.show()

def controlador(kP, kI, kD, sys):
    Cr = 1e-6
    Re = 1 / (kI * Cr)
    Rr = kP * Re
    Ce = kD / Rr

    numPID = [Re * Rr * Ce * Cr, (Re * Ce + Rr * Cr), 1]
    denPID = [Re * Cr, 0]

    PID = ctrl.tf(numPID, denPID)
    X = ctrl.series(PID, sys)
    sysPID = ctrl.feedback(X, 1, sign=-1)
    return sysPID

hipoPID = controlador(1.558, 373.621, 0.000532, syshipo)
print("Función de transferencia del hipotenso en lazo cerrado:")
print(hipoPID)

hiperPID = controlador(13.294, 397.820, 0.0346, syshiper)
print("Función de transferencia del hipertenso en lazo cerrado:")
print(hiperPID)

_, PID1 = ctrl.forced_response(hipoPID, T=t, U=Pp1, X0=x0)
_, PID2 = ctrl.forced_response(hiperPID, T=t, U=Pp1, X0=x0)

def plotsignals(t, Pp1, Pp2, Pp3, PID1, PID2):
    plt.rcParams['font.family'] = 'Times New Roman'

    colors = np.array([
        [194, 180, 234],
        [249, 207, 221],
        [177, 223, 243]
    ]) / 255.0

    fig, axs = plt.subplots(2, 1, figsize=(8, 6.2), facecolor='white')

    axs[0].plot(t, Pp1, '-', linewidth=1.2, color=colors[0], label='Pp(t): Normotenso')
    axs[0].plot(t, Pp2, '-', linewidth=1.2, color=colors[1], label='Pp(t): Hipotenso')
    axs[0].plot(t, PID1, ':', linewidth=2.0, color=colors[2], label='PID(t): Hipotenso')

    y1_all = np.concatenate([Pp1, Pp2, PID1])
    y1min, y1max = np.min(y1_all), np.max(y1_all)
    y1margin = 0.08 * (y1max - y1min if y1max != y1min else 1)

    axs[0].set_xlim(t[0], t[-1])
    axs[0].set_ylim(y1min - y1margin, y1max + y1margin)
    axs[0].set_xticks(np.arange(0, 16, 1))
    axs[0].margins(x=0.01)
    axs[0].set_xlabel('t [s]', fontsize=11)
    axs[0].set_ylabel('Vi(t) [V]', fontsize=11)
    axs[0].set_title('Normotenso vs Hipotenso', fontsize=11)
    axs[0].grid(True, alpha=0.25)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=9)
    axs[0].tick_params(labelsize=10)

    axs[1].plot(t, Pp1, '-', linewidth=1.2, color=colors[0], label='Pp(t): Normotenso')
    axs[1].plot(t, Pp3, '-', linewidth=1.2, color=colors[1], label='Pp(t): Hipertenso')
    axs[1].plot(t, PID2, ':', linewidth=2.0, color=colors[2], label='PID(t): Hipertenso')

    y2_all = np.concatenate([Pp1, Pp3, PID2])
    y2min, y2max = np.min(y2_all), np.max(y2_all)
    y2margin = 0.08 * (y2max - y2min if y2max != y2min else 1)

    axs[1].set_xlim(t[0], t[-1])
    axs[1].set_ylim(y2min - y2margin, y2max + y2margin)
    axs[1].set_xticks(np.arange(0, 16, 1))
    axs[1].margins(x=0.01)
    axs[1].set_xlabel('t [s]', fontsize=11)
    axs[1].set_ylabel('Vi(t) [V]', fontsize=11)
    axs[1].set_title('Normotenso vs Hipertenso', fontsize=11)
    axs[1].grid(True, alpha=0.25)
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=9)
    axs[1].tick_params(labelsize=10)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.08, hspace=0.65)
    plt.savefig('Cardiovascular.pdf', format='pdf')
    plt.show()

plotsignals(t, Pp1, Pp2, Pp3, PID1, PID2)