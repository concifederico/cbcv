import wx
import globvar

cuadro = 0

from frames import VideoFrame

""" Standard way of starting a wxPython app """
app = wx.App(False)
frame = VideoFrame(None)

frame.Maximize(True)
frame.Show()
app.MainLoop()
