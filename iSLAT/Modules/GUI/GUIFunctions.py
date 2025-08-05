import tkinter as tk
from tkinter import ttk
from .Tooltips import CreateToolTip

def create_button(frame, theme, text, command, row, column):
        btn_theme = theme["buttons"].get(
            text.replace(" ", ""), theme["buttons"]["DefaultBotton"]
        )
        
        btn = tk.Button(
            frame, text=text,
            command=command
        )
        btn.grid(row=row, column=column, padx=2, pady=2, sticky="nsew")
        CreateToolTip(btn, f"{text} button")
        return btn

def create_menu_btn(frame, theme, text, row, column):
        btn_theme = theme["buttons"].get(
            text.replace(" ", ""), theme["buttons"]["DefaultBotton"]
        )
        drpdwn = tk.Menubutton(
            frame, text=text,
            relief=tk.RAISED, 
        )
        drpdwn.grid(row=row, column=column, padx=2, pady=2, sticky="nsew")
        return drpdwn

def create_scrollable_frame(parent, height = 150, width = 300, vertical = False, horizontal = False, **kwargs):
        
        canvas_frame = ttk.Frame(parent)
        canvas_frame.grid(row=0, column=0, sticky="nsew")

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
        canvasscroll.create_window((0, 0), window=data_frame, anchor="nw")

        data_frame.bind("<Configure>", 
                        lambda event: canvasscroll.configure(scrollregion=canvasscroll.bbox("all"))
                        )
        return data_frame

def create_wrapper_frame(parent, row, col, bg = "darkgrey", sticky = "nsew", columnspan = 1) -> tk.Frame:
        wrapper = tk.Frame(parent, borderwidth=1, relief="flat", bg=bg)
        wrapper.grid(row = row, column=col, sticky= sticky, columnspan=columnspan)
        wrapper.grid_rowconfigure(0, weight=1)
        wrapper.grid_columnconfigure(0, weight=1)

        return wrapper


        