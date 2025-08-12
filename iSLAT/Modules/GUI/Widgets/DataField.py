import tkinter as tk
from tkinter import ttk
from .ResizableFrame import ResizableFrame

class DataField(ttk.Frame):
    """
    A class to represent a text area in a GUI application.
    Now inherits from ResizableFrame for consolidated theming.
    """
    def __init__(self, name: str, value: any, master: tk.Widget, theme=None):
        # Initialize the ResizableFrame with theme
        super().__init__(master)
        
        self.name = name
        self.value = value

        self.label_frame = tk.LabelFrame(self, relief="solid", borderwidth=1, text=self.name)
        self.label_frame.pack(fill="both", pady=0)

        self.label_frame.rowconfigure(0, weight=1)
        self.label_frame.columnconfigure(0, weight=1)
        

        # Text widget with scrollbar
        self.text = tk.Text(self.label_frame, height=13, width=24, wrap="word")
        self.scrollbar = ttk.Scrollbar(self.label_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        # Place text & scrollbar
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def insert_text(self, content, clear_first=True, console_print = False):
        if clear_first:
            self.clear()

        if console_print:
            print(content)

        self.text.insert("end", str(content) + "\n")
        self.text.see("end")

    def clear(self):
        self.text.delete("1.0", "end")

    def delete(self, start="1.0", end="end"):
        self.text.delete(start, end)

    def __repr__(self):
        return f"DataField(name={self.name}, value={self.value})"