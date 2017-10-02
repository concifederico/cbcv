#frames.py

from __future__ import division

import wx
import gui
import globvar
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import csv
import os

from datetime import datetime

from monitor import analizar

from models import WebcamFeed

"""
The Controller and View class.
It creates an openCV object and on each update retrieves a webcam image from it.
It will then draw the webcam image onto the frame.
"""

class VideoFrame(gui.wxVideoFrame):
    def __init__(self, parent):
        self.parent = parent
        gui.wxVideoFrame.__init__(self, parent)
        ww, hw = self.m_panelVideo.GetSize()
        if globvar.VideoWebCam == True:
            """ Create the webcam feed/openCV object  """
            self.webcam = WebcamFeed()
            if not self.webcam.has_webcam():
                print 'Webcam has not been detected.'
                self.Close()
            else:
                """ Sets the size based on the webcam image size """
                h, w = self.webcam.size()
                globvar.iniciovideo = True

        else:
            globvar.cap = cv2.VideoCapture(self.text_ctrl_archivo.GetValue().encode('utf-8'))
            if (globvar.cap.isOpened()):
                ret, frame = globvar.cap.read()
                w, h, _ = frame.shape
                globvar.iniciovideo = True
        #if globvar.iniciovideo == True:

        self.m_panelVideo.SetSize((int(hw / float(h) * w), hw))
        self.window_1.SplitVertically(self.window_1_pane_1, self.window_1_pane_2, int(hw / float(h) * w))

        #self.SetSize(wx.Size(w, h))

        """ Bind custom paint events """
        self.m_panelVideo.Bind(wx.EVT_ERASE_BACKGROUND, self.onEraseBackground)
        self.m_panelVideo.Bind(wx.EVT_PAINT, self.onPaint)

        """ Bind a custom close event (needed for Windows) """
        self.Bind(wx.EVT_CLOSE, self.onClose)

        """ App states """
        self.STATE_RUNNING = 1
        self.STATE_CLOSING = 2
        self.state = self.STATE_RUNNING

    """ When closing, timer needs to be stopped and frame destroyed """

    def onClose(self, event):
        if not self.state == self.STATE_CLOSING:
            self.state = self.STATE_CLOSING
            self.timer.Stop()
            self.Destroy()

    """ Main Update loop that calls the Paint function """

    def onUpdate(self, event):
        if self.state == self.STATE_RUNNING:
            self.Refresh()

    """ Retrieves a new webcam image and paints it onto the frame """

    def onPaint(self, event):
        fw, fh = self.m_panelVideo.GetSize()

        # Retrieve a scaled image from the opencv model
        if globvar.iniciovideo == True:
            if globvar.VideoWebCam == True:
                frame = self.webcam.get_image(1280, 720)
            else:
                if (globvar.cap.isOpened()):
                    ret, frame = globvar.cap.read()
                    if ret == False:
                        globvar.cap = cv2.VideoCapture(self.text_ctrl_archivo.GetValue().encode('utf-8'))
                        ret, frame = globvar.cap.read()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame.shape[:2]

                    #frame = np.zeros((h, w, 3), np.uint8)

            if self.radio_btn_90.GetValue() == 1:
                frame = imutils.rotate_bound(frame, 90)
            if self.radio_btn_180.GetValue() == 1:
                frame = imutils.rotate_bound(frame, 180)
            if self.radio_btn_270.GetValue() == 1:
                frame = imutils.rotate_bound(frame, 270)

            #cv2.rectangle(frame , (50, 25), (100, 50), (0, 0, 255), 15)

            if globvar.inicioanalisis == True:
                diferencia, largo, diametro, angulo, progreso, frame = analizar(frame, self.slider_hmin.GetValue(),self.slider_hmax.GetValue(),self.slider_smin.GetValue(),self.slider_smax.GetValue(),self.slider_vmin.GetValue(),self.slider_vmax.GetValue(),self.spin_ctrl_filas.GetValue(),self.slider_desperado.GetValue(),self.slider_prefiltro.GetValue(),self.slider_laterali.GetValue(),self.slider_laterals.GetValue(),self.spin_ctrl_anchobanda.GetValue(), self.combo_box_tiempo.GetValue(), self.combo_box_Objeto.GetValue(), self.slider_calib.GetValue(), self.spin_button_display.GetValue(), self.text_ctrl_vidoverlay.GetValue(), self.spin_button_display.GetValue(), self.checkbox_AdvOverlay.GetValue())
                self.gauge_1.SetValue(int(progreso))
                if globvar.flaghist == True:

                    if globvar.Produciendo == True:
                        pLSE = (globvar.LMedidoA > globvar.LSE).sum() / float((globvar.LMedidoA > 0).sum()) * 100
                        pLIE =  (globvar.LMedidoA < globvar.LIE).sum() / float((globvar.LMedidoA > 0).sum()) * 100
                        pOK = 100 - pLSE - pLIE
                    else:
                        pLSE = 0
                        pLIE = 0
                        pOK = 0

                    self.label_diametrolse.LabelText = "{:.0f}".format(pLSE) + "%"

                    self.label_diametrolie.LabelText = "{:.0f}".format(pLIE) + "%"

                    self.label_diametrooptimo.LabelText = "{:.0f}".format(pOK) + "%"

                    '''
                    TS
                    ------

                    Fecha-Hora
                    Objeto tipo
                    Largo
                    Diametro
                    Proporcion LIE
                    Proporcion LSE
                    Produciendo
                    
                    '''

                    globvar.TS[globvar.TSi, 0] = datetime.now().replace(microsecond=0)
                    globvar.TS[globvar.TSi, 1] = self.combo_box_Objeto.GetValue()[:-1]
                    globvar.TS[globvar.TSi, 2] = globvar.LMedido
                    globvar.TS[globvar.TSi, 3] = globvar.DMedido
                    globvar.TS[globvar.TSi, 4] = pLIE
                    globvar.TS[globvar.TSi, 5] = pLSE
                    globvar.TS[globvar.TSi, 6] = globvar.Produciendo
                    globvar.TSi = globvar.TSi + 1

                    '''
                            Campos
                            ------
    
                            Fecha-Hora
                            Objeto tipo
                            Objetos detectadas
                            Largo
                            Diametro
                            Angulo
                            '''

                    globvar.Ciclos = globvar.Ciclos + 1

                    if self.spin_button_registro.GetValue() == globvar.Ciclos:

                        globvar.Ciclos = 0

                        if self.checkbox_csv.GetValue() == True:

                            with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "db.csv","a") as f:
                                writer = csv.writer(f)
                                fields = [str(datetime.now().replace(microsecond=0)), self.combo_box_Objeto.GetValue(), globvar.DMedidoC, "%.1f" % globvar.LMedido, "%.1f" % globvar.DMedido, "%.1f" % globvar.AMedido]
                                writer.writerow(fields)
                            f.close

                            with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "dbh.csv","a") as f:
                                writer = csv.writer(f)
                                fields = [str(datetime.now().replace(microsecond=0)), self.combo_box_Objeto.GetValue()[:-1], np.round(globvar.DMedidoA,1), np.round(globvar.LMedidoA,1)]
                                writer.writerow(fields)
                            f.close

                            with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "dbf.csv","a") as f:
                                writer = csv.writer(f)
                                fields = [str(datetime.now().replace(microsecond=0)), self.combo_box_Objeto.GetValue()[:-1], np.round(globvar.filas[1:self.spin_ctrl_filas.GetValue()+1],1)]
                                writer.writerow(fields)
                            f.close

                        #csv = np.genfromtxt('dbh.csv', delimiter=",")

                    #csv = csv[(csv[:,1] == self.combo_box_Objeto.GetValue())]

                    with open(os.path.realpath(__file__)[:-len(os.path.basename(os.path.realpath(__file__)))] + "dbh.csv", "rb") as f:
                        data_iter = csv.reader(f, delimiter=",",quotechar='"')
                        data = [data for data in data_iter]
                    f.close

                    data_array = np.asarray(data)

                    data_array[:,2] = self.combo_box_Objeto.GetValue()

                    self.label_diametro.LabelText = "{:.1f}".format(diametro)

                    if diametro < globvar.LIE:
                        self.label_diametro.SetForegroundColour(wx.Colour(255, 127, 0))
                    elif diametro > globvar.LSE:
                        self.label_diametro.SetForegroundColour(wx.Colour(255, 0, 0))
                    else:
                        self.label_diametro.SetForegroundColour(wx.Colour(0, 127, 0))

                    self.label_largo.LabelText = "{:.1f}".format(largo)

                    if largo < globvar.LIE:
                        self.label_largo.SetForegroundColour(wx.Colour(255, 127, 0))
                    elif diametro > globvar.LSE:
                        self.label_largo.SetForegroundColour(wx.Colour(255, 0, 0))
                    else:
                        self.label_largo.SetForegroundColour(wx.Colour(0, 127, 0))

                    self.label_FPS.LabelText = "{:.1f}".format(globvar.FFPS/globvar.TFPS)

                    self.label_diferencia.LabelText = "{:.1f}".format(diferencia)

                    self.label_angulo.LabelText = "{:.0f}".format(angulo)

                    # Plot Diametro
                    self.axes_sd.clear()

                    self.axes_sd.plot(globvar.TS[0:globvar.TSi, 0].astype(datetime), globvar.TS[0:globvar.TSi, 3].astype(float), label='Diametro')
                    self.axes_sd.plot(globvar.TS[0:globvar.TSi, 0].astype(datetime), globvar.TS[0:globvar.TSi, 2].astype(float), label='Largo')
                    self.axes_sd.legend(framealpha=0.5)
                    self.axes_sd.set_xlabel('Hora')
                    self.axes_sd.set_ylabel('Diametro (mm)')
                    self.axes_sd.set_title('Dimensiones - Tiempo')
                    self.axes_sd.autoscale(enable=True, axis='y')
                    #
                    for tick in self.axes_sd.get_xticklabels():
                        tick.set_rotation(45)

                    self.canvas_sd.draw()

                    # Plot Fracciones
                    self.axes_st.clear()

                    self.axes_st.stackplot(globvar.TS[0:globvar.TSi, 0].astype(datetime), 100 - globvar.TS[0:globvar.TSi, 4].astype(float) - globvar.TS[0:globvar.TSi, 5].astype(float), globvar.TS[0:globvar.TSi, 4].astype(float), globvar.TS[0:globvar.TSi, 5].astype(float), colors=['g','orange','r'], labels=['OK','<LIE','>LSE'])
                    self.axes_st.legend(framealpha=0.5)
                    self.axes_st.set_xlabel('Hora')
                    self.axes_st.set_ylabel('Fraccion %')
                    self.axes_st.set_title('Tolerancias - Tiempo')
                    for tick in self.axes_st.get_xticklabels():
                        tick.set_rotation(45)

                    self.canvas_st.draw()

                    self.axes_f.clear()
                    globvar.filas = np.nan_to_num(globvar.filas)
                    barras = self.axes_f.bar(np.arange(self.spin_ctrl_filas.GetValue()),
                                             globvar.filas[1:self.spin_ctrl_filas.GetValue() + 1], 0.75, color='r')
                    self.axes_f.set_xlabel('Filas')
                    self.axes_f.set_ylabel('Diametro (mm)')
                    self.axes_f.set_title(r'Diametro - Filas')

                    self.canvas_f.draw()

                    # Hist Largo

                    self.axes_l.clear()
                    n, bins, patches = self.axes_l.hist(globvar.LMedidoA, normed=True, facecolor='green',
                                                        alpha=0.7)
                    self.axes_l.set_xlabel('Largo (mm)')
                    self.axes_l.set_ylabel('Densidad')
                    self.axes_l.set_title(r'Histograma - Largo')

                    twentyfifth, seventyfifth = 50, 58

                    for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
                        if rightside < twentyfifth:
                            patch.set_facecolor('orange')
                        elif leftside > seventyfifth:
                            patch.set_facecolor('red')

                    self.canvas_l.draw()

                    # Hist Diametro
                    self.axes_d.clear()
                    n, bins, patches = self.axes_d.hist(globvar.DMedidoA,normed = True, facecolor='green', alpha=0.7)
                    self.axes_d.set_xlabel('Diametro (mm)')
                    self.axes_d.set_ylabel('Densidad')
                    self.axes_d.set_title(r'Histograma - Diametro')

                    twentyfifth, seventyfifth = 50,58

                    for patch, rightside, leftside in zip(patches, bins[1:], bins[:-1]):
                        if rightside < twentyfifth:
                            patch.set_facecolor('orange')
                        elif leftside > seventyfifth:
                            patch.set_facecolor('red')

                    self.canvas_d.draw()

                    # Bar Filas
                    self.axes_f.clear()
                    globvar.filas = np.nan_to_num(globvar.filas)
                    barras = self.axes_f.bar(np.arange(self.spin_ctrl_filas.GetValue()), globvar.filas[1:self.spin_ctrl_filas.GetValue()+1], 0.75, color='r')
                    self.axes_f.set_xlabel('Filas')
                    self.axes_f.set_ylabel('Diametro (mm)')
                    self.axes_f.set_title(r'Diametro - Filas')
                    self.axes_f.set_ylim(globvar.LIE / 2, globvar.LSE * 1.5)

                    self.canvas_f.draw()

                    self.axes_f.clear()
                    globvar.filas = np.nan_to_num(globvar.filas)
                    barras = self.axes_f.bar(np.arange(self.spin_ctrl_filas.GetValue()),
                                             globvar.filas[1:self.spin_ctrl_filas.GetValue() + 1], 0.75, color='r')
                    self.axes_f.set_xlabel('Filas')
                    self.axes_f.set_ylabel('Diametro (mm)')
                    self.axes_f.set_title(r'Diametro - Filas')

                    self.canvas_f.draw()

                    # Bar Filas
                    self.axes_f.clear()
                    globvar.filas = np.nan_to_num(globvar.filas)
                    barras = self.axes_f.bar(np.arange(self.spin_ctrl_filas.GetValue()), globvar.filas[1:self.spin_ctrl_filas.GetValue()+1], 0.75, color='r')
                    self.axes_f.set_xlabel('Filas')
                    self.axes_f.set_ylabel('Diametro (mm)')
                    self.axes_f.set_title(r'Diametro - Filas')

                    self.canvas_f.draw()

                    self.axes_f.clear()
                    globvar.filas = np.nan_to_num(globvar.filas)
                    barras = self.axes_f.bar(np.arange(self.spin_ctrl_filas.GetValue()),
                                             globvar.filas[1:self.spin_ctrl_filas.GetValue() + 1], 0.75, color='r')
                    self.axes_f.set_xlabel('Filas')
                    self.axes_f.set_ylabel('Diametro (mm)')
                    self.axes_f.set_title(r'Diametro - Filas')

                    self.canvas_f.draw()

                    globvar.AMedidoS = 0
                    globvar.DifMedidoS = 0
                    globvar.LMedidoS = 0
                    globvar.DMedidoS = 0
                    globvar.DMedidoC = 0
                    globvar.DMedidoA = np.empty(1)
                    globvar.LMedidoA = np.empty(1)
                    globvar.FMedidoA = np.empty(1)
                    globvar.filasS = np.zeros(shape=(20, 1))
                    globvar.filasC = np.zeros(shape=(20, 1))

                    globvar.flaghist = False

            #frame = cv2.resize(frame, (200,200))

            h, w = frame.shape[:2]
            ww,hw = self.window_1_pane_1.GetSize()
            self.m_panelVideo.SetSize((int(hw/float(h)*w),hw))
            self.window_1_pane_1.SetMinSize((int(hw/float(h)*w),hw))


            h1,w1 = self.window_1_pane_1.GetSize()

            if h >0 and w > 0:
                frame = cv2.resize(frame, (h1,w1))
                h, w = frame.shape[:2]
            h2, w2 = frame.shape[:2]

                #self.m_panelVideo.SetSize((405, 720))
            #self.window_1_pane_1.SetSize((h, w))

            image = wx.Bitmap.FromBuffer(w, h, frame)

            # Use Buffered Painting to avoid flickering
            dc = wx.BufferedPaintDC(self.m_panelVideo)
            dc.DrawBitmap(image, 0, 0)

    """ Background will never be erased, this avoids flickering """

    def onEraseBackground(self, event):
        return
