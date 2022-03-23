import wx

def ask(parent=None, message='', default_value=''):


    dlg = wx.TextEntryDialog(parent, message)
    dlg.ShowModal()
    result = dlg.GetValue()
    dlg.Destroy()
    return result

# Initialize wx App
app = wx.App()
app.MainLoop()

# Call Dialog
w = ask(message='What is your name?')
print ('Your name was', w)
x = ask(message='What is your address?')
print ('Your name was', x)
y = ask(message='What is your name?')
print ('Your name was', y)
z = ask(message='What is your name?')
print ('Your name was', z)