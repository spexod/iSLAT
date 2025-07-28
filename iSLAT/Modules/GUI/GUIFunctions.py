import tkinter as tk
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