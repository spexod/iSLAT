from tkinter import Toplevel, Label, LEFT, SOLID  # For ttk.Style

# Module-level theme reference for tooltip colors.
# Call set_tooltip_theme(theme_dict) once at GUI startup so every
# tooltip automatically uses the current theme colours.
_current_theme = None

_LIGHT_DEFAULTS = {"tooltip_background": "peachpuff", "tooltip_foreground": "#000000"}
_DARK_DEFAULTS  = {"tooltip_background": "#3C3F41",  "tooltip_foreground": "#F0F0F0"}


def set_tooltip_theme(theme: dict | None) -> None:
    """Set the module-level theme used by all future tooltips."""
    global _current_theme
    _current_theme = theme


def _resolve_tooltip_colors(explicit_bg=None, explicit_fg=None):
    """Return (bg, fg) for a tooltip, respecting explicit overrides,
    the current theme, and sensible fallback defaults."""
    theme = _current_theme or {}
    bg = explicit_bg or theme.get("tooltip_background") or _LIGHT_DEFAULTS["tooltip_background"]
    fg = explicit_fg or theme.get("tooltip_foreground") or _LIGHT_DEFAULTS["tooltip_foreground"]
    return bg, fg


class ToolTip(object):
    def __init__(self, widget, text, bg=None, fg=None):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.active = True
        self.x = self.y = 0
        # Store explicit overrides; resolved at show-time so theme
        # changes that happen after construction are picked up.
        self._explicit_bg = bg
        self._explicit_fg = fg

    def showtip(self):
        "Display text in tooltip window"
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 45
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        bg, fg = _resolve_tooltip_colors(self._explicit_bg, self._explicit_fg)
        label = Label(tw, text=self.text, justify=LEFT,
                       background=bg, foreground=fg,
                       relief=SOLID, borderwidth=1,
                       font=("tahoma", "12", "normal"))
        label.pack(ipadx=1)

    def disable(self):
        self.active = False

    def enable(self):
        self.active = True

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def CreateToolTip(widget, text, bg=None, fg=None):
    toolTip = ToolTip(widget, text, bg=bg, fg=fg)

    def enter(event):
        if toolTip.active:
            toolTip.showtip()

    def leave(event):
        if toolTip.active:
            toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

    return toolTip