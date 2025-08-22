from tkinter import Toplevel, Label, LEFT, SOLID  # For ttk.Style

class ToolTip(object):
    def __init__(self, widget, text, bg = "peachpuff"):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.active = True
        self.x = self.y = 0
        self.bg: str = bg

    def showtip(self):
        "Display text in tooltip window"
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 45
        y = y + cy + self.widget.winfo_rooty () + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                       background=self.bg, relief=SOLID, borderwidth=1,
                       font=("tahoma", "12", "normal"))
        label.pack (ipadx=1)

    def disable(self):
        self.active = False

    def enable(self):
        self.active = True

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text, bg = "peachpuff"):
    toolTip = ToolTip(widget, text, bg=bg)

    def enter(event):
        if toolTip.active:
            toolTip.showtip()

    def leave(event):
        if toolTip.active:
            toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

    return toolTip