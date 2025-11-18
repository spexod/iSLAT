import tkinter as tk
from tkinter import ttk, font
from .Tooltips import CreateToolTip

def create_button(frame, theme, text, command, row, column, tip_text = None):
        btn_theme = theme["buttons"].get(
            text.replace(" ", ""), theme["buttons"]["DefaultBotton"]
        )

        btn = tk.Button(
            frame, text=text,
            command=command,
        )

        if tip_text: 
            CreateToolTip(btn, tip_text)

        btn.grid(row=row, column=column, padx=1, pady=2, sticky="nsew")
        return btn

def create_menu_btn(frame, theme, text, row, column):
        drpdwn = tk.Menubutton(
            frame, text=text,
            relief=tk.RAISED, 
        )
        drpdwn.grid(row=row, column=column, padx=1, pady=2, sticky="nsew")
        return drpdwn

def create_scrollable_frame(parent, height = 150, width = 300, vertical = False, horizontal = False, row=0, col = 0):
        
        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=row, column=col, sticky="nsew")

        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(col, weight=1)

        canvasscroll = tk.Canvas(canvas_frame, height=height, width=width, highlightthickness=0)
        canvasscroll.grid(row=0, column=0, sticky="nsew")
        if vertical is True:
                vscrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvasscroll.yview)
                vscrollbar.grid(row=0, column=1, sticky="ns")
                canvasscroll.configure(yscrollcommand=vscrollbar.set)
        if horizontal is True:
                hscrollbar = tk.Scrollbar(canvas_frame, orient="horizontal", command=canvasscroll.xview)
                hscrollbar.grid(row=1, column=0, columnspan= 2, sticky="ew")
                canvasscroll.configure(xscrollcommand=hscrollbar.set)
        
        # Allow resizing of the canvasscroll and outer_frame
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        data_frame = ttk.Frame(canvasscroll)
        window_item = canvasscroll.create_window((0, 0), window=data_frame, anchor="nw")

        data_frame.bind("<Configure>", 
                        lambda event: canvasscroll.configure(scrollregion=canvasscroll.bbox("all"))
                        )
        # this is attempt to add default parameters button 
        # def resize_frame(event):
        #     canvasscroll.itemconfig(window_item, width=event.width, height=event.height)

        # canvasscroll.bind("<Configure>", resize_frame)
        
        return data_frame

def create_wrapper_frame(parent, row, col, bg = "darkgrey", sticky = "nsew", columnspan = 1) -> tk.Frame:
        wrapper = tk.Frame(parent, borderwidth=1, relief="flat", bg=bg)
        wrapper.grid(row = row, column=col, sticky= sticky, columnspan=columnspan)
        wrapper.grid_rowconfigure(0, weight=1)
        wrapper.grid_columnconfigure(0, weight=1)

        return wrapper

class ColorButton(tk.Label):
    def __init__(self, master, color="#4CAF50", command=None, **kwargs):
        if color is None:
            color = "#4CAF50"
        super().__init__(master, bg=color, width=2, height=1,relief="flat", **kwargs)
        self.color = self._to_hex_color(color)
        self.command = command
        self.clicked = False

        # Compute darker shade for click effect
        self.darker_color = self._darken_color(self.color, 0.7)
        self.hover_color = self._darken_color(self.color, 0.9)

        self.config(highlightthickness=1, highlightbackground="#ccc")

        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<ButtonRelease-1>", self.on_release)

    def _to_hex_color(self, color):
                # Convert any Tk color name or hex string to hex string
        try:
        # If color is already hex, just normalize and return
                if color.startswith("#") and (len(color) == 7 or len(color) == 4):
                        return color.lower()
        except Exception:
                pass

        # Otherwise, convert color name to hex via winfo_rgb
        r, g, b = self.winfo_rgb(color)
        r = int(r / 65535 * 255)
        g = int(g / 65535 * 255)
        b = int(b / 65535 * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _darken_color(self, hex_color, factor):
        """Darken color by factor (0 < factor < 1)."""
        hex_color = hex_color.lstrip("#")
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        dark_rgb = tuple(max(int(c * factor), 0) for c in rgb)
        return "#%02x%02x%02x" % dark_rgb

    def on_enter(self, event):
        if not self.clicked:
            self.config(bg=self.hover_color)

    def on_leave(self, event):
        if not self.clicked:
            self.config(bg=self.color)

    def on_press(self, event):
        self.config(bg=self.darker_color, relief="sunken")

    def on_release(self, event):
        self.config(bg=self.hover_color if self._is_inside(event) else self.color,
                    relief="flat")
        if self._is_inside(event) and self.command:
            self.command()
    def _is_inside(self, event):
        x, y = event.x, event.y
        return 0 <= x < self.winfo_width() and 0 <= y < self.winfo_height()
    
    def change_color(self, color):
        color = self._to_hex_color(color)
        self.color = color
        # self.config(bg = color)

        self.darker_color = self._darken_color(self.color, 0.7)
        self.hover_color = self._darken_color(self.color, 0.9)

    def add_command(self, command=None):
        self.command = command