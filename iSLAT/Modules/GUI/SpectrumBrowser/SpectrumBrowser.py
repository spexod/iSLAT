import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import spexod.API 

class SpectrumBrowser:
    def __init__(self, master, theme):
        self.master = master
        self.theme = theme
        self.master.title("Spectrum Browser")

        self.all_spectra = spexod.API.get_spectra()
        self.save_all_spectra_to_json()  # Save all spectra to JSON file
        #print("self.all_spectra", self.all_spectra)
        #print("First spectrum:", self.all_spectra[0] if self.all_spectra else "No spectra available")

        self.selected_spectrum = None  # Will hold the selected spectrum object

        #self.create_scrollable_frame()
        #self.populate_spectra()

    def save_all_spectra_to_json(self, filename="all_spectra.json"):
        import json
        spectra_list = []
        for spectrum in self.all_spectra:
            # Copy all data from the spectrum dictionary
            spectrum_data = dict(spectrum)
            # Optionally, add wavelength and flux arrays if not already present
            if "wavelengths" not in spectrum_data:
                spectrum_data["wavelengths"] = spexod.API.get_wavelengths(spectrum["spectrum_handle"])
            if "fluxes" not in spectrum_data:
                spectrum_data["fluxes"] = spexod.API.get_fluxes(spectrum["spectrum_handle"])
            spectra_list.append(spectrum_data)
        with open(filename, 'w') as f:
            json.dump(spectra_list, f, indent=4)
        print(f"All spectra saved to {filename}.")

    def create_scrollable_frame(self):
        self.canvas = tk.Canvas(self.master, bg=self.theme["background"])
        self.scrollbar = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def populate_spectra(self):
        for idx, spectrum in enumerate(self.all_spectra):
            fig = plt.Figure(figsize=(4, 2), dpi=100)
            ax = fig.add_subplot(111)
            
            wavelength = spexod.API.get_wavelengths(spectrum["spectrum_handle"])
            fluxes = spexod.API.get_fluxes(spectrum["spectrum_handle"])
            flux = fluxes[0]

            ax.plot(wavelength, flux, color=self.theme["foreground"])
            ax.set_title(f"Spectrum {idx+1}")
            ax.set_facecolor(self.theme["background"])
            ax.tick_params(axis='x', colors=self.theme["foreground"])
            ax.tick_params(axis='y', colors=self.theme["foreground"])
            ax.spines['bottom'].set_color(self.theme["foreground"])
            ax.spines['left'].set_color(self.theme["foreground"])
            ax.spines['right'].set_color(self.theme["foreground"])
            ax.spines['top'].set_color(self.theme["foreground"])
            
            canvas = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
            canvas.get_tk_widget().pack(pady=5)
            
            # Make it clickable by binding with lambda capturing the spectrum
            canvas.get_tk_widget().bind("<Button-1>", lambda e, s=spectrum: self.on_spectrum_click(s))

    def on_spectrum_click(self, spectrum):
        self.selected_spectrum = spectrum
        print(f"Selected spectrum: {spectrum}")

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    theme = {
        "background": "#1e1e1e",
        "foreground": "#d4d4d4"
    }
    app = SpectrumBrowser(root, theme)
    root.mainloop()