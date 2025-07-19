import tkinter as tk
from tkinter import ttk

class DataField:
    """
    A class to represent a text area in a GUI application.
    No layout here — the caller does grid/pack for .frame.
    """
    def __init__(self, name: str, value: any, master: tk.Widget):
        self.name = name
        self.value = value

        # Entire frame to hold label + text + scrollbar
        self.frame = ttk.Frame(master)

        # Label
        self.label = ttk.Label(self.frame, text=self.name)
        self.label.pack(fill="x")

        # Text widget with scrollbar
        self.text = tk.Text(self.frame, height=20, width=60, wrap="word")
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=self.scrollbar.set)

        # Place text & scrollbar
        self.text.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        self._apply_basic_theming()
    
    def _apply_basic_theming(self):
        """Apply basic theming to the text widget"""
        try:
            # Use stored theme if available, otherwise try to get from parent chain
            theme = getattr(self, '_theme', None)
            
            if not theme:
                parent = self.frame.master
                while parent and not theme:
                    if hasattr(parent, 'theme'):
                        theme = parent.theme
                        break
                    elif hasattr(parent, 'master'):
                        parent = parent.master
                    else:
                        break
            
            if theme:
                self.text.configure(
                    bg=theme.get("data_field_background", "#23272A"),
                    fg=theme.get("foreground", "#F0F0F0"),
                    insertbackground=theme.get("foreground", "#F0F0F0"),
                    selectbackground=theme.get("selection_color", "#00FF99"),
                    selectforeground=theme.get("background", "#181A1B")
                )
                
                # Configure TTK style for scrollbar
                try:
                    style = ttk.Style()
                    style.configure("DataField.Vertical.TScrollbar",
                                  background=theme.get("background_accent_color", "#23272A"),
                                  troughcolor=theme.get("background", "#181A1B"),
                                  bordercolor=theme.get("background_accent_color", "#23272A"),
                                  arrowcolor=theme.get("foreground", "#F0F0F0"),
                                  darkcolor=theme.get("background_accent_color", "#23272A"),
                                  lightcolor=theme.get("background_accent_color", "#23272A"))
                    
                    self.scrollbar.configure(style="DataField.Vertical.TScrollbar")
                except:
                    pass
        except:
            pass

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

    def apply_theme(self, theme=None):
        """Public method to apply theme to the data field"""
        if theme:
            # Store theme for future use
            self._theme = theme
            
        # Apply theming
        self._apply_basic_theming()

    def __repr__(self):
        return f"DataField(name={self.name}, value={self.value})"