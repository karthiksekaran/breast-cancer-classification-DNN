import wx
import os

class DNN(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, "Breast Cancer Classification System", size=(1650,1000))
        panel = wx.Panel(self)
        wx.StaticText(panel, -1, "Breast Cancer Classification", (650, 10))
        ran = wx.Button(panel, label='Classify', pos=(200, 300), size=(200, 60))
        ler = wx.Button(panel, label='Show Training Data', pos=(500, 300), size=(200, 60))
        tes = wx.Button(panel, label='Show Testing Data', pos=(800, 300), size=(200, 60))
        cls = wx.Button(panel, label='Click Here To Close Window', pos=(1100, 300), size=(200, 60))

        self.Bind(wx.EVT_BUTTON, self.ran, ran)
        self.Bind(wx.EVT_BUTTON, self.cls, cls)
        self.Bind(wx.EVT_BUTTON, self.tes, tes)
        self.Bind(wx.EVT_BUTTON, self.ler, ler)

    def ran(self, event):
     os.system(r" python C:\Final-Review\Files\Deep-Neural-Network.py")

    def cls(self, event):
        self.Destroy()

    def ler(self, event):
        os.system(r" python E:\Project\Final-Review\Files\train-read.py")
    def tes(self, event):
        os.system(r" python E:\Project\Final-Review\Files\test-read.py")

    def closewindow(self, event):
        self.Destroy()


if __name__ == '__main__':
    App = wx.App()
    frame = DNN(parent=None, id=-1)
    frame.Show()
    App.MainLoop()

