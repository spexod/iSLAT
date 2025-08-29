import tkinter as tk
from tkinter import ttk
from iSLAT.Modules.Hitran_data import download_hitran_data

class MoleculeSelector:
    def __init__(self, master, data_field):
        self.master = master
        self.data_field = data_field  # Reference to the data field in the main GUI
        self.mols = []
        self.basem = []
        self.isot = []

        # Isotopologue data with assigned numbers
        self.isotopologue_data = {
            "H2O": [("H216O", 1), ("H218O", 2), ("H217O", 3), ("HD16O", 4), ("HD18O", 5), ("HD17O", 6), ("D216O", 7)],
            "CO2": [("12C16O2", 1), ("13C16O2", 2), ("16O12C18O", 3), ("16O12C17O", 4), ("16O13C18O", 5), ("16O13C17O", 6), 
                    ("12C18O2", 7), ("17O12C18O", 8), ("12C17O2", 9), ("13C18O2", 10), ("18O13C17O", 11), ("13C17O2", 12)],
            "O3": [("16O3", 1), ("16O16O18O", 2), ("16O18O16O", 3), ("16O16O17O", 4), ("16O17O16O", 5)],
            "N2O": [("14N216O", 1), ("14N15N16O", 2), ("15N14N16O", 3), ("14N218O", 4), ("14N217O", 5)],
            "CO": [("12C16O", 1), ("13C16O", 2), ("12C18O", 3), ("12C17O", 4), ("13C18O", 5), ("13C17O", 6)],
            "CH4": [("12CH4", 1), ("13CH4", 2), ("12CH3D", 3), ("13CH3D", 4)],
            "O2": [("16O2", 1), ("16O18O", 2), ("16O17O", 3)],
            "NO": [("14N16O", 1), ("15N16O", 2), ("14N18O", 3)],
            "SO2": [("32S16O2", 1), ("34S16O2", 2), ("33S16O2", 3), ("16O32S18O", 4)],
            "NO2": [("14N16O2", 1), ("15N16O2", 2)],
            "NH3": [("14NH3", 1), ("15NH3", 2)],
            "HNO3": [("H14N16O3", 1), ("H15N16O3", 2)],
            "OH": [("16OH", 1), ("18OH", 2), ("16OD", 3)],
            "HF": [("H19F", 1), ("D19F", 2)],
            "HCl": [("H35Cl", 1), ("H37Cl", 2), ("D35Cl", 3), ("D37Cl", 4)],
            "HBr": [("H79Br", 1), ("H81Br", 2), ("D79Br", 3), ("D81Br", 4)],
            "HI": [("H127I", 1), ("D127I", 2)],
            "ClO": [("35Cl16O", 1), ("37Cl16O", 2)],
            "OCS": [("16O12C32S", 1), ("16O12C34S", 2), ("16O13C32S", 3), ("16O12C33S", 4), ("18O12C32S", 5), ("16O13C34S", 6)],
            "H2CO": [("H212C16O", 1), ("H213C16O", 2), ("H212C18O", 3)],
            "HOCl": [("H16O35Cl", 1), ("H16O37Cl", 2)],
            "N2": [("14N2", 1), ("14N15N", 2)],
            "HCN": [("H12C14N", 1), ("H13C14N", 2), ("H12C15N", 3)],
            "CH3Cl": [("12CH335Cl", 1), ("12CH337Cl", 2)],
            "H2O2": [("H216O2", 1)],
            "C2H2": [("12C2H2", 1), ("H12C13CH", 2), ("H12C12CD", 3)],
            "C2H6": [("12C2H6", 1), ("12CH313CH3", 2)],
            "PH3": [("31PH3", 1)],
            "COF2": [("12C16O19F2", 1), ("13C16O19F2", 2)],
            "SF6": [("32S19F6", 1)],
            "H2S": [("H232S", 1), ("H234S", 2), ("H233S", 3)],
            "HCOOH": [("H12C16O16OH", 1)],
            "HO2": [("H16O2", 1)],
            "O": [("16O", 1)],
            "ClONO2": [("35Cl16O14N16O2", 1), ("37Cl16O14N16O2", 2)],
            "NO+": [("14N16O+", 1)],
            "HOBr": [("H16O79Br", 1), ("H16O81Br", 2)],
            "C2H4": [("12C2H4", 1), ("12CH213CH2", 2)],
            "CH3OH": [("12CH316OH", 1)],
            "CH3Br": [("12CH379Br", 1), ("12CH381Br", 2)],
            "CH3CN": [("12CH312C14N", 1)],
            "CF4": [("12C19F4", 1)],
            "C4H2": [("12C4H2", 1)],
            "HC3N": [("H12C314N", 1)],
            "H2": [("H2", 1), ("HD", 2)],
            "CS": [("12C32S", 1), ("12C34S", 2), ("13C32S", 3), ("12C33S", 4)],
            "SO3": [("32S16O3", 1)],
            "C2N2": [("12C214N2", 1)],
            "COCl2": [("12C16O35Cl2", 1), ("12C16O35Cl37Cl", 2)],
            "SO": [("32S16O", 1), ("34S16O", 2), ("32S18O", 3)],
            "CH3F": [("12CH319F", 1)],
            "GeH4": [("74GeH4", 1), ("72GeH4", 2), ("70GeH4", 3), ("73GeH4", 4), ("76GeH4", 5)],
            "CS2": [("12C32S2", 1), ("32S12C34S", 2), ("32S12C33S", 3), ("13C32S2", 4)],
            "CH3I": [("12CH3127I", 1)],
            "NF3": [("14N19F3", 1)]
        }


        self.window = tk.Toplevel(master)
        self.window.title("Download from HITRAN")

        self.frame = ttk.Frame(self.window)
        self.frame.pack(padx=10, pady=10)

        self.molecule_label = ttk.Label(self.frame, text="Molecule:")
        self.molecule_label.grid(row=0, column=0, padx=5, pady=5)

        self.molecule_var = tk.StringVar()
        self.molecule_combobox = ttk.Combobox(self.frame, textvariable=self.molecule_var)
        self.molecule_combobox['values'] = list(self.isotopologue_data.keys())
        self.molecule_combobox.grid(row=0, column=1, padx=5, pady=5)
        self.molecule_combobox.bind('<<ComboboxSelected>>', self.update_isotopologues)

        self.isotopologue_label = ttk.Label(self.frame, text="Isotopologues:")
        self.isotopologue_label.grid(row=1, column=0, padx=5, pady=5)

        self.isotopologue_var = tk.StringVar()
        self.isotopologue_combobox = ttk.Combobox(self.frame, textvariable=self.isotopologue_var)
        self.isotopologue_combobox.grid(row=1, column=1, padx=5, pady=5)

        self.add_button = ttk.Button(self.frame, text="Download HITRAN file", command=self.add_molecule)
        self.add_button.grid(row=2, column=0, columnspan=2, pady=10)

        self.done_button = ttk.Button(self.frame, text="Close Window", command=self.on_done)
        self.done_button.grid(row=3, column=0, columnspan=2, pady=10)

    def update_isotopologues(self, event):
        print("updating isotopologues")
        selected_molecule = self.molecule_var.get()
        isotopologues = self.isotopologue_data.get(selected_molecule, [])
        self.isotopologue_combobox['values'] = [iso[0] for iso in isotopologues]
        if isotopologues:
            self.isotopologue_var.set(isotopologues[0][0])
        else:
            self.isotopologue_var.set("")

    def add_molecule(self):
        print("adding molecule")
        mol = self.molecule_var.get()
        isotopologue = self.isotopologue_var.get()
        isotopologue_list = self.isotopologue_data.get(mol, [])
        isotope = next((num for iso, num in isotopologue_list if iso == isotopologue), None)

        if isotope is not None:
            self.mols.append(mol)
            self.basem.append(isotopologue)
            self.isot.append(isotope)
            

            missed_mols = download_hitran_data(self.basem, self.mols, self.isot)
            if missed_mols:
                for (bm, mol, iso) in missed_mols:
                    error_message = f"Could not load Molecule: {bm}, Isotopologue: {mol}, Isotope Number: {iso}"
                    self.data_field.insert_text(error_message, clear_after=False, console_print=True)
                    return
            

            print(f"Added Molecule: {mol}, Isotopologue: {isotopologue}, Isotope Number: {isotope}")

            # Update the main GUI data_field
            self.data_field.delete('1.0', "end")
            self.data_field.insert_text(f"{isotopologue} downloaded from HITRAN.", clear_after=True, console_print=True)

    def on_done(self):
        self.window.destroy()