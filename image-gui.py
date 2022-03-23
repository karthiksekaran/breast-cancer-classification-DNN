import wx
import os

class DNN(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, "Breast Cancer Classification System", size=(1650,1000))
        panel = wx.Panel(self)
        wx.StaticText(panel, -1, "Breast Cancer Classification", (1200, 10))
        pic = wx.StaticBitmap(panel)
        pic.SetBitmap(wx.Bitmap("image.jpg"))
        self.ShowFullScreen(True)
        ran = wx.Button(panel, label='Proceed', pos=(1000, 400), size=(200, 60))
        cls = wx.Button(panel, label='Exit', pos=(1300, 400), size=(200, 60))

        self.Bind(wx.EVT_BUTTON, self.ran, ran)
        self.Bind(wx.EVT_BUTTON, self.cls, cls)

    def ran(self, event):
     os.system(r" python C:\Final-Review\Files\about.py")

    def cls(self, event):
        self.Destroy()

    def closewindow(self, event):
        self.Destroy()


if __name__ == '__main__':
    App = wx.App()
    frame = DNN(parent=None, id=-1)
    frame.Show()
    App.MainLoop()

