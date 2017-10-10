# globvar.py

import numpy as np

def init():

    global VideoWebCam
    VideoWebCam = False

    global TimerStatus
    TimerStatus = False

    global size

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
    gecant = np.zeros(shape=(400, ))

    #TimeSeries
    global TS
    TS = np.empty(100000, dtype=[('Fecha','datetime64[s]'), ('Objeto','S10'), ('Largo','float'), ('Diametro','float'), ('LIE','float'), ('LSE','float'), ('Produciendo','int')])
    global TSi #Index
    TSi = 0

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
    AMedido = 0
    global numfilas
    numfilas = 16

    global filas
    filas = np.zeros(shape=(20, 1))
    global filasS
    filasS = np.zeros(shape=(20, 1))
    global filasC
    filasC = np.zeros(shape=(20, 1))

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
    global LMedidoS
    LMedidoS = 0
    global LMedido
    LMedido = 0
    global IniciaralInicio
    IniciaralInicio = False
    global Capturar
    Capturar = False

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











