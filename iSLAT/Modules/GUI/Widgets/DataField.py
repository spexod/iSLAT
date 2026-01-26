import tkinter as tk
from tkinter import ttk
from .ResizableFrame import ResizableFrame

class DataField(ttk.Frame):
    """
    A class to represent a text area in a GUI application.
    Now inherits from ResizableFrame for consolidated theming.
    """
    def __init__(self, value: any, master: tk.Widget, theme=None):
        # Initialize the ResizableFrame with theme
        super().__init__(master)
        
        self.value = value
        self.clear_before_next = False

        self.label_frame = tk.LabelFrame(self, relief="solid", borderwidth=1, text="iSLAT Text Output")
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
        
    def insert_text(self, content, clear_after=True, console_print = False):
        try:
            # Check if the widget still exists and is valid
            if not hasattr(self, 'text') or not self.text.winfo_exists():
                return
            
            if self.clear_before_next:
                self.clear()
                
            # if clear_after:
            #     self.clear()

            if console_print:
                print(content)

            self.text.insert("end", str(content) + "\n")
            self.text.see("end")

            if clear_after:
                self.clear_before_next = True
            else:
                self.clear_before_next = False
        except Exception as e:
            # Silently handle GUI destruction errors
            if console_print:
                print(content)  # At least print to console if GUI is destroyed

    def clear(self):
        try:
            # Check if the widget still exists and is valid
            if hasattr(self, 'text') and self.text.winfo_exists():
                self.text.delete("1.0", "end")
        except Exception as e:
            # Silently handle GUI destruction errors
            pass

    def delete(self, start="1.0", end="end"):
        self.text.delete(start, end)

    def __repr__(self):
        return f"DataField(name={self.name}, value={self.value})"