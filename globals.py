# config.py

import numpy as np

global VideoWebCam
VideoWebCam = False

global TimerStatus
TimerStatus = False

global cuadro
cuadro = 0
global tic
global videoframe
global cap
global fps
fps = 0
global inicioanalisis
inicioanalisis = False
global iniciovideo
iniciovideo = False
global gecant
gecant = np.zeros(shape=(400, 16))
global Tiempo
Tiempo = 0
global nGalleta
nGalleta = 0
global flagfilas
flagfilas = False
global flaghist
flaghist = False
global Fresultados
Fresultados = np.zeros(shape=(60*8, 20))
global AMedido
global AMedidoS
AMedidoS = 0
global DifMedido
DifMedido = 0
global DifMedidoS
DifMedidoS = 0
global DMedidoA
DMedidoA = np.array([]).reshape(0,1)
global LMedidoA
LMedidoA = np.array([]).reshape(0,1)
global FMedidoA
FMedidoA = np.array([]).reshape(0,1)
global Produciendo
Produciendo = False
global DMedidoS
DMedidoS = 0
global DMedidoC
DMedidoC = 0
global DMedido
DMedido = 0
global LIE
global LSE
global MaxDiff
global TRefresh
global Ciclos
Ciclos = 0
global Dispositivo

global TFPS
TFPS = 0
global FFPS
FFPS = 0











