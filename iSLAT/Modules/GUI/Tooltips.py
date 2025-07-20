from tkinter import Toplevel, Label, LEFT, SOLID  # For ttk.Style

class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty () + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = Label(tw, text=self.text, justify=LEFT,
                       background="peachpuff", relief=SOLID, borderwidth=1,
                       font=("tahoma", "12", "normal"))
        label.pack (ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)

    def enter(event):
        toolTip.showtip(text)

    def leave(event):
        toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

'''
CreateToolTip(import_button, text='Query HITRAN.org\n'
                                   'database to download\n'
                                   'new molecules')

CreateToolTip(addmol_button, text='Add molecule to\n'
                                   'the GUI from files\n'
                                   'downloaded from HITRAN')

CreateToolTip(saveparams_button, text='Save current molecules\n'
                                       'and their parameters\n'
                                       'for this input datafile')

CreateToolTip(loadparams_button, text='Load previously saved\n'
                                       'molecules and parameters\n'
                                       'for this input datafile')

CreateToolTip(defmol_button, text='Load default\n'
                                   'molecule list\n'
                                   'into the GUI')

CreateToolTip(export_button, text='Export current\n'
                                   'models into csv files')

CreateToolTip(toggle_button, text='Turn legend on/off')

CreateToolTip(file_button, text='Select input spectrum datafile')

CreateToolTip(linefile_button, text='Select input line list\n'
                                     'from those available in iSLAT\n'
                                     'or previously saved by the user')

CreateToolTip(linesave_button, text='Select output file\n'
                                     'to save line measurements\n'
                                     'with "Save Line" or "Fit Saved Lines"')

# ---------------------------------------------------------
CreateToolTip(xp1_entry, text='Start wavelength\n'
                               'for the upper plot\n'
                               'units: μm')

CreateToolTip(rng_entry, text='Wavelength range\n'
                               'for the upper plot\n'
                               'units: μm')

CreateToolTip(min_lamb_entry, text='Minimum wavelength\n'
                                    'to calculate the models\n'
                                    'units: μm')

CreateToolTip(max_lamb_entry, text='Maximum wavelength\n'
                                    'to calculate the models\n'
                                    'units: μm')

CreateToolTip(dist_entry, text='Distance to the\n'
                                'observed target\n'
                                'units: pc')

CreateToolTip(star_rv_entry, text='Radial velocity (helioc.)\n'
                                   'of the observed target\n'
                                   'units: km/s')

CreateToolTip(fwhm_entry, text='FWHM for convolution\n'
                                'of the model spectra\n'
                                'units: km/s')

CreateToolTip(intrinsic_line_width_entry, text='Line broadening (FWHM)\n'
                                                '(thermal/turbulence)\n'
                                                'units: km/s')

CreateToolTip(specsep_entry, text='Separation threshold\n'
                                   'for "Find Single Lines"\n'
                                   'units: μm')

CreateToolTip(spandropd, text='Select molecule for line inspection,\n'
                               'the population diagram, and all the\n'
                               'molecule-specific functions')

CreateToolTip(fwhmtolerance_entry, text='Line broadening tolerance\n'
                                         'in de-blender\n'
                                         'units: km/s')

CreateToolTip(centrtolerance_entry, text='Line centroid tolerance\n'
                                   'in de-blender\n'
                                   'units: μm')

# ------------------------------------------------------------------------
CreateToolTip(save_button, text='Save strongest line\n'
                                 'from the current line inspection\n'
                                 'into the "Output Line Measurements"')

CreateToolTip(fit_button, text='Fit line currently\n'
                                'selected for line inspection')

CreateToolTip(slabfit_button, text='Fit single slab model for molecule\n'
                                    'selected in the drop-down menu\n'
                                    'using flux measurements from input')

CreateToolTip(deblender_button, text='De-blend selected feature\n'
                                    'using a multi-gaussian fit\n'
                                    'with tolerance values above')

CreateToolTip(savedline_button, text='Show saved lines\n'
                                      'from the "Input Line List"')

CreateToolTip(fitsavedline_button, text='Fit all lines from the "Input Line List"\n'
                                         'and save into the "Output Line Measurements"')

CreateToolTip(autofind_button, text='Find single lines\n'
                                     'using separation threshold\n'
                                     'set in the "Line Separ."')

CreateToolTip(atomlines_button, text='Show atomic lines\n'
                                      'from the available line list')'''