#!/usr/bin/python

# CSv1.py2

'''
20 Filas de Objetos
20 Columnas
= 400 filas de array

= 10 columnas de array
'''

# import the necessary packages

import numpy as np
import imutils
import cv2
import time
import os
import csv
import globvar
import gui

from datetime import datetime, timedelta

#from scipy.spatial import distance as dist

import matplotlib
matplotlib.use('WXAgg')

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure

# import xlsxwriter

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

    ''' 
    20 Filas de Objetos
    20 Columnas
    = 400 filas de array

    0 X
    1 Y
    2 Diametro
    3 SumaDiametro
    4 SumaD
    5 Orientacion
    6 Ocupado
    7 Ignorar
    8 Tiempo
    9 Fila
    10 Diferencia XY
    11 SumaDiferencia XY
    12 Largo
    13 SumaLargo
    14 Angulo
    15 SumaAngulo

    = 10 columnas de array
    '''

def dist(x,y):
    return np.linalg.norm(x-y)

def analizar(img, hmin, hmax, smin, smax, vmin, vmax, numfilas, DEsperado, Prefiltro, laterali, laterals, anchobanda, seltiempo, Objeto, calib, TRefresh, Texto, Graficos,AdvOverlay):

    import math
    flagorden = False

    fields = ['first', 'second', 'third']

    #gecant = np.zeros(shape=(400, 10))

    if globvar.cuadro == 0:
        filas = []
        columnas = []
        #tic = time.clock()

    globvar.cuadro = globvar.cuadro + 1

    #globvar.tic = time.clock()

    height, width = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    HSV_MIN = np.array([hmin, smin, vmin], np.uint8)
    HSV_MAX = np.array([hmax, smax, vmax], np.uint8)

    frame_threshed = cv2.inRange(hsv, HSV_MIN, HSV_MAX)

    frame_threshed = cv2.dilate(frame_threshed, (100, 100), iterations=2)

    cnts = np.empty(1)

    if globvar.RPi == True:
        _ , cnts, _ = cv2.findContours(frame_threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        cnts, _ = cv2.findContours(frame_threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    orig = img.copy()

    if globvar.Tiempo <= datetime.now().replace(microsecond=0) - timedelta(seconds=TRefresh):
        globvar.Tiempo = datetime.now().replace(microsecond=0)

        globvar.flaghist = True

        if globvar.DMedidoC > 0 :
            globvar.AMedido = globvar.AMedidoS / globvar.DMedidoC
            globvar.DMedido = globvar.DMedidoS / globvar.DMedidoC
            globvar.DifMedido = globvar.DifMedidoS / globvar.DMedidoC
            globvar.LMedido = globvar.LMedidoS / globvar.DMedidoC

            globvar.filas[1:numfilas+1] = globvar.filasS[1:numfilas+1] / globvar.filasC[1:numfilas+1]



            if globvar.Produciendo == False:
                globvar.Produciendo = True

                with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "Log.csv",
                          "a") as f:
                    writer = csv.writer(f)
                    fields = [str(datetime.now().replace(microsecond=0)), "Largando"]
                    writer.writerow(fields)
                    f.close

        else:
            globvar.AMedido = 0
            globvar.DMedido = 0
            globvar.DifMedido = 0
            globvar.LMedido = 0

            if globvar.Produciendo == True:
                globvar.Produciendo = False

                with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "Log.csv",
                          "a") as f:
                    writer = csv.writer(f)
                    fields = [str(datetime.now().replace(microsecond=0)), "Cortando"]
                    writer.writerow(fields)
                    f.close

    i = 0
    # loop over the contours individually

    suma = 0
    div = 0
    gec = np.zeros(shape=(400, numfilas))

    if len(cnts) > 0:

	for c in cnts:
	    if cv2.contourArea(c) > Prefiltro:

	        suma = suma + cv2.contourArea(c)
	        div = div + 1
	if div > 0:
	    prom = suma / div
	else:
	    prom = 0

	        #fconts = []  # filtrados contornos
	        #fcontsnd = []  # filtrados contornos sin Objetos dobles

	ADimD = 0  # Promedio del diametro
	d = 0


	for c in cnts:

	    # if the contour is not sufficiently large, ignore it
	    if cv2.contourArea(c) > .4 * prom and cv2.contourArea(c) < 4 * prom:

	       box = cv2.minAreaRect(c)
	       box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	       box = np.array(box, dtype="int")

	       (tl, tr, br, bl) = box
	       (tltrX, tltrY) = midpoint(tl, tr)
	       (blbrX, blbrY) = midpoint(bl, br)
	       (tlblX, tlblY) = midpoint(tl, bl)
	       (trbrX, trbrY) = midpoint(tr, br)

	       # compute the Euclidean distance between the midpoints
	       dA = dist(np.array(tltrX, tltrY), np.array(blbrX, blbrY))
	       dB = dist(np.array(tlblX, tlblY), np.array(trbrX, trbrY))

	       (cgX, cgY) = midpoint(tl, br)  # centro de la Objeto

	       if cgX > DEsperado and (width - cgX) > DEsperado:

	                #fconts.append(c)
	                #if dB / dA > 1.75 or dB / dA < 0.75:
	                #    fcontsnd.append(c)

	            pixelsPerMetric = anchobanda / float(height - laterali - laterals) * (1 + calib / float(1000))

	                # compute the size of the object
	            dimA = dA * pixelsPerMetric
	            dimB = dB * pixelsPerMetric

	            gec[d, 0] = cgX
	            gec[d, 1] = cgY
	            gec[d, 2] = math.sqrt(cv2.contourArea(c) / 3.14159) * 2 * pixelsPerMetric
	            gec[d, 6] = True
	            gec[d, 10] = (max(dA,dB)-min(dA,dB))* pixelsPerMetric
	            gec[d, 12] = max(dA, dB) * pixelsPerMetric

	            if dA >= dB and abs(tltrY-blbrY) > 0:
	                gec[d, 14] = 90 - math.atan(abs((tltrX-blbrX)/(tltrY-blbrY))) / float(math.pi) * 180
	            else:
	                if (tlblY-trbrY) > 0:
	                    gec[d, 14] = 90 - math.atan(abs((tlblX-trbrX)/(tlblY-trbrY))) / float(math.pi) * 180

	                # gec[d,3] = gec[d:2] + dimD
	            d = d + 1

	                # ++++++++++++ Printing +++++++++++++++

	            cv2.drawContours(orig, c, -1, (0, 255, 0), 2)

	                # draw the midpoints on the img
	            if AdvOverlay == True:
	                #if dA > dB:
	                    #cv2.circle(orig, (int(tltrX), int(tltrY)), 2, (255, 0, 255), -1)
	                    #cv2.circle(orig, (int(blbrX), int(blbrY)), 2, (255, 0, 255), -1)
	                    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (0, 0, 255), 2)
	                #else:
	                    #cv2.circle(orig, (int(tlblX), int(tlblY)), 2, (255, 0, 255), -1)
	                    #cv2.circle(orig, (int(trbrX), int(trbrY)), 2, (255, 0, 255), -1)
	                    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (0, 255, 0), 2)

	                # draw the object sizes on the img
	                # cv2.putText(orig, "{:.0f}mm".format(dimD), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if AdvOverlay == True:
        cv2.putText(orig, Texto, (int(100), int(20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.line(orig, (int(DEsperado),int(0)), (int(DEsperado),int(height)), (127, 127, 255), 2)
        cv2.line(orig, (int(width - DEsperado),int(0)), (int(width - DEsperado), int(height)), (127, 127, 255), 2)

        cv2.line(orig, (int(DEsperado),int(laterali)), (int(width - DEsperado), int(laterali)), (127, 255, 0), 2)
        cv2.line(orig, (int(width/2),int(0)), (int(width/2), int(laterali)), (127, 255, 0), 2)

        cv2.line(orig, (int(DEsperado),int(height-laterals)), (int(width - DEsperado), int(height-laterals)), (127, 255, 0), 2)
        cv2.line(orig, (int(width/2),int(height)), (int(width/2), int(height-laterals)), (127, 255, 0), 2)

    # Linkeo de Objetos en Cuadro anterior vs el Nuevo Cuadro

    gec = gec[gec[:, 0] != 0]

    globvar.flagfilas = False

    if not (np.all(globvar.gecant == 0)):

        gec = gec[np.argsort(gec[:, 0])]

        for ant in globvar.gecant:
            p = closest_node([ant[0], ant[1]], gec[:, [0, 1]])
            if ant[7] == False:
                # No ignorar esta Objeto
                if abs(gec[p, 0]-ant[0]) > 0:
                    dydx = abs((gec[p, 1] - ant[1]) / (gec[p, 0]-ant[0] ))
                else:
                    dydx = 0
                dista =  math.sqrt(pow((gec[p, 1] - ant[1]), 2) + pow((gec[p, 0]-ant[0] ), 2))
                if dista < DEsperado and gec[p, 0] > ant[0]:
                    # Si el angulo de avance de la Objeto dY/dX < 1
                    gec[p, 3] = ant[3] + gec[p, 2]
                    gec[p, 4] = ant[4] + 1
                    if ant[8] == 0:
                        gec[p, 8] = time.clock()
                    else:
                        gec[p, 8] = ant[8]

                    gec[p, 7] = ant[7]
                    gec[p, 9] = ant[9]
                    gec[p, 11] = ant[11] + gec[p, 10]
                    gec[p, 13] = ant[13] + gec[p, 12]
                    gec[p, 15] = ant[15] + gec[p, 14]

                    if ant[9] == 0 and gec.shape[0] > 2*numfilas and globvar.flagfilas == False:
                        if np.mean(gec[numfilas+1:numfilas*2+1,0]) - (np.max(gec[numfilas+1:numfilas*2+1,0])-np.min(gec[numfilas+1:numfilas*2+1,0])) *.5 > np.mean(gec[0:numfilas,0]) + (np.max(gec[0:numfilas,0])-np.min(gec[0:numfilas,0])) *.5:
                            globvar.flagfilas = True
                            gs = gec[0:16,]
                            gs = gs[np.argsort(gs[:, 1])]

                            for f in range(1, numfilas+1):
                                gs[f-1, 9] = f

                            gec = np.vstack((gs, gec[numfilas:,]))
                    if AdvOverlay == True:
                        cv2.line(orig, (int(ant[0]), int(ant[1])), (int(gec[p, 0]), int(gec[p, 1])), (255, 50, 255),2)
                    #cv2.putText(orig, "{:.0f}mm".format(gec[p, 3] / gec[p, 4]), (int(gec[p, 0] + 30), int(gec[p, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    if gec[p, 7] == False and AdvOverlay == True:
                        if gec[p, 9] > 0:
                            cv2.putText(orig, "{:.0f}".format(gec[p, 9]), (int(gec[p, 0] + 35), int(gec[p, 1]) - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        resultado = gec[p, 13] / float(gec[p, 4])
                        if resultado < globvar.LIE:
                            cv2.putText(orig, "{:.1f}".format(resultado) , (int(gec[p, 0] + 35), int(gec[p, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 127, 0), 1)
                        elif resultado > globvar.LSE:
                            cv2.putText(orig, "{:.1f}".format(resultado), (int(gec[p, 0] + 35), int(gec[p, 1])),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                        else:
                            cv2.putText(orig, "{:.1f}".format(resultado), (int(gec[p, 0] + 35), int(gec[p, 1])),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                        #cv2.putText(orig, "{:.0f}".format(gec[p, 14]), (int(gec[p, 0] + 25), int(gec[p, 1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        #cv2.putText(orig, "{:.1f}".format(time.clock()-gec[p, 8]), (int(gec[p, 0] + 25), int(gec[p, 1])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                else:
                    # No ignorar esta Objeto, pero ahora su tracking se perdio y hay que ver porque?
                    if ant[0] > (width - DEsperado * 1.25 ):  # Porque esta saliendo por arriba?

                        if ant[7] == False and ant[4] > 4:
                            #print "Saliendo por borde"
                            #globvar.AMedidoS = globvar.AMedidoS + float(ant[9])
                            globvar.DMedidoS = globvar.DMedidoS + ant[3] / float(ant[4])
                            globvar.DMedidoC = globvar.DMedidoC + 1
                            globvar.DMedidoA = np.append(globvar.DMedidoA, ant[3] / float(ant[4]))
                            globvar.LMedidoA = np.append(globvar.LMedidoA, ant[13] / float(ant[4]))
                            globvar.FMedidoA = np.append(globvar.FMedidoA, ant[13] / float(ant[4]))

                            if ant[9] > 0:
                                globvar.filasS[int(ant[9])] = globvar.filasS[int(ant[9])] + ant[3] / float(ant[4])
                                globvar.filasC[int(ant[9])] = globvar.filasC[int(ant[9])] + 1

                            globvar.DifMedidoS = globvar.DifMedidoS + ant[11] / float(ant[4])
                            globvar.LMedidoS = globvar.LMedidoS + ant[13] / float(ant[4])
                            globvar.AMedidoS = globvar.AMedidoS + ant[15] / float(ant[4])
                            globvar.TFPS = globvar.TFPS + time.clock()-gec[p, 8]
                            globvar.FFPS = globvar.FFPS + ant[4]

                    else:
                        if AdvOverlay == True:
                            cv2.putText(orig, "E",(int(gec[p, 0]), int(gec[p, 1])),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                        #print "Saliendo por error con " + str(dista) +  " / " + str(dydx)
            else:
                gec[p, 9] = ant[9]

    globvar.gecant = gec

    Progress = min(max(int((datetime.now().replace(microsecond=0) - globvar.Tiempo).seconds / float(TRefresh) * 100),0),100)

    return globvar.DifMedido, globvar.LMedido, globvar.DMedido, globvar.AMedido, Progress, orig


