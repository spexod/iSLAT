import tkinter as tk
from tkinter import ttk

class ResizableFrame(tk.Frame):
    """A frame that can be resized by dragging its borders and provides consolidated theming."""
    
    def __init__(self, parent, orientation='vertical', sash_size=4, theme=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.orientation = orientation
        self.sash_size = sash_size
        self.frames = []
        self.sashes = []
        self.dragging = False
        self.drag_data = {"x": 0, "y": 0, "sash": None}
        self.total_weight = 0
        self.initialized = False
        self.theme = theme or {}
        
        # Apply initial theme
        self._apply_base_theme()
        
    def add_frame(self, frame, weight=1, minsize=50, dynamic_minsize=False):
        """Add a frame to the resizable container."""
        frame_info = {
            'frame': frame, 
            'weight': weight, 
            'minsize': minsize,
            'dynamic_minsize': dynamic_minsize,  # Whether to calculate minsize based on content
            'current_size': 0  # Will be calculated in _calculate_initial_sizes
        }
        self.frames.append(frame_info)
        self.total_weight += weight
        
        # Create sash if not the first frame
        if len(self.frames) > 1:
            sash = tk.Frame(self, cursor='sb_v_double_arrow' if self.orientation == 'vertical' else 'sb_h_double_arrow')
            # Style the sash to make it more visible
            sash.configure(bg='#888888', relief='raised', bd=1)
            self.sashes.append(sash)
            self._bind_sash_events(sash, len(self.sashes) - 1)
        
        # Delay initial layout until after the widget is mapped
        self.after(10, self._initialize_layout)
        
        # Bind to configure events to handle window resizing
        self.bind("<Configure>", self._on_configure)
    
    def _apply_theme_to_sashes(self, theme):
        """Apply theme styling to sashes."""
        for sash in self.sashes:
            sash.configure(bg=theme.get("toolbar", "#888888"))
            # Update hover colors too
            def make_hover_handlers(s, theme_ref):
                return (
                    lambda e: s.configure(bg=theme_ref.get("selection_color", "#999999")),
                    lambda e: s.configure(bg=theme_ref.get("toolbar", "#888888"))
                )
            enter_handler, leave_handler = make_hover_handlers(sash, theme)
            sash.bind("<Enter>", enter_handler)
            sash.bind("<Leave>", leave_handler)
    
    def _calculate_content_minsize(self, frame):
        """Calculate minimum size based on frame content."""
        try:
            # Update the frame to ensure all widgets have been placed
            frame.update_idletasks()
            
            if self.orientation == 'vertical':
                # For vertical orientation, calculate required height
                return self._calculate_required_height(frame)
            else:
                # For horizontal orientation, calculate required width
                return self._calculate_required_width(frame)
        except tk.TclError:
            # If frame isn't ready, return a reasonable default
            return 50
    
    def _calculate_required_height(self, frame):
        """Calculate the minimum height required for all widgets in a frame."""
        min_height = 0
        
        # Check all child widgets
        for child in frame.winfo_children():
            try:
                child.update_idletasks()
                
                # Get widget geometry
                child_height = child.winfo_reqheight()
                child_y = child.winfo_y()
                
                # Calculate total space needed (position + height + padding)
                if child.winfo_manager() == 'pack':
                    # For packed widgets, add some padding
                    padx, pady = self._get_pack_padding(child)
                    total_height = child_y + child_height + pady
                elif child.winfo_manager() == 'grid':
                    # For grid widgets, consider row span
                    grid_info = child.grid_info()
                    pady = grid_info.get('pady', 0)
                    if isinstance(pady, tuple):
                        pady = sum(pady)
                    total_height = child_y + child_height + pady
                else:
                    total_height = child_y + child_height + 10  # Default padding
                
                min_height = max(min_height, total_height)
                
            except tk.TclError:
                continue
        
        # Add some extra padding for the frame itself
        return max(min_height + 20, 50)  # Minimum of 50 pixels
    
    def _calculate_required_width(self, frame):
        """Calculate the minimum width required for all widgets in a frame."""
        min_width = 0
        
        # Check all child widgets
        for child in frame.winfo_children():
            try:
                child.update_idletasks()
                
                # Get widget geometry
                child_width = child.winfo_reqwidth()
                child_x = child.winfo_x()
                
                # Calculate total space needed (position + width + padding)
                if child.winfo_manager() == 'pack':
                    # For packed widgets, add some padding
                    padx, pady = self._get_pack_padding(child)
                    total_width = child_x + child_width + padx
                elif child.winfo_manager() == 'grid':
                    # For grid widgets, consider column span
                    grid_info = child.grid_info()
                    padx = grid_info.get('padx', 0)
                    if isinstance(padx, tuple):
                        padx = sum(padx)
                    total_width = child_x + child_width + padx
                else:
                    total_width = child_x + child_width + 10  # Default padding
                
                min_width = max(min_width, total_width)
                
            except tk.TclError:
                continue
        
        # Add some extra padding for the frame itself
        return max(min_width + 20, 50)  # Minimum of 50 pixels
    
    def _get_pack_padding(self, widget):
        """Get padding information for a packed widget."""
        try:
            pack_info = widget.pack_info()
            padx = pack_info.get('padx', 0)
            pady = pack_info.get('pady', 0)
            
            # Handle tuple padding values
            if isinstance(padx, tuple):
                padx = sum(padx)
            if isinstance(pady, tuple):
                pady = sum(pady)
                
            return padx, pady
        except tk.TclError:
            return 10, 10  # Default padding
    
    def _on_configure(self, event):
        """Handle window resize events."""
        if event.widget == self and self.initialized:
            self._calculate_initial_sizes()
            self._layout_frames()
    
    def _initialize_layout(self):
        """Initialize the layout after the widget is mapped."""
        if not self.initialized and self.winfo_width() > 1 and self.winfo_height() > 1:
            self._update_dynamic_minsizes()
            self._calculate_initial_sizes()
            self._layout_frames()
            self.initialized = True
        elif not self.initialized:
            # Try again later if widget isn't ready
            self.after(50, self._initialize_layout)
    
    def _update_dynamic_minsizes(self):
        """Update dynamic minimum sizes for all frames that need it."""
        for frame_info in self.frames:
            if frame_info['dynamic_minsize']:
                calculated_minsize = self._calculate_content_minsize(frame_info['frame'])
                frame_info['minsize'] = calculated_minsize
    
    def _calculate_initial_sizes(self):
        """Calculate initial sizes for frames based on their weights."""
        if self.orientation == 'vertical':
            available_space = self.winfo_height() - (len(self.sashes) * self.sash_size)
        else:
            available_space = self.winfo_width() - (len(self.sashes) * self.sash_size)
        
        # Ensure minimum space
        total_minsize = sum(frame['minsize'] for frame in self.frames)
        available_space = max(available_space, total_minsize)
        
        # First, allocate minimum sizes for zero-weight frames
        remaining_space = available_space
        for frame_info in self.frames:
            if frame_info['weight'] == 0:
                frame_info['current_size'] = frame_info['minsize']
                remaining_space -= frame_info['minsize']
            else:
                remaining_space -= frame_info['minsize']
        
        # Then distribute remaining space among weighted frames
        total_weight = sum(frame['weight'] for frame in self.frames if frame['weight'] > 0)
        
        if total_weight > 0 and remaining_space > 0:
            for frame_info in self.frames:
                if frame_info['weight'] > 0:
                    proportional_size = int((frame_info['weight'] / total_weight) * remaining_space)
                    frame_info['current_size'] = frame_info['minsize'] + proportional_size
    
    def update_dynamic_sizes(self):
        """Update dynamic minimum sizes and recalculate layout."""
        self._update_dynamic_minsizes()
        self._calculate_initial_sizes()
        self._layout_frames()
    
    def set_frame_dynamic_minsize(self, frame, dynamic=True):
        """Enable or disable dynamic minimum sizing for a specific frame."""
        for frame_info in self.frames:
            if frame_info['frame'] == frame:
                frame_info['dynamic_minsize'] = dynamic
                if dynamic:
                    # Immediately recalculate if enabling dynamic sizing
                    self.after_idle(self.update_dynamic_sizes)
                break
    
    def _bind_sash_events(self, sash, sash_index):
        """Bind mouse events to sash for dragging."""
        sash.bind("<Button-1>", lambda e: self._start_drag(e, sash_index))
        sash.bind("<B1-Motion>", lambda e: self._on_drag(e, sash_index))
        sash.bind("<ButtonRelease-1>", lambda e: self._end_drag(e, sash_index))
        
    def _start_drag(self, event, sash_index):
        """Start dragging a sash."""
        self.dragging = True
        self.drag_data["sash"] = sash_index
        if self.orientation == 'vertical':
            self.drag_data["y"] = event.y_root
        else:
            self.drag_data["x"] = event.x_root
    
    def _on_drag(self, event, sash_index):
        """Handle sash dragging."""
        if not self.dragging:
            return
            
        if self.orientation == 'vertical':
            delta = event.y_root - self.drag_data["y"]
            self.drag_data["y"] = event.y_root
        else:
            delta = event.x_root - self.drag_data["x"]
            self.drag_data["x"] = event.x_root
        
        self._resize_frames(sash_index, delta)
    
    def _end_drag(self, event, sash_index):
        """End dragging."""
        self.dragging = False
        
    def _resize_frames(self, sash_index, delta):
        """Resize frames based on sash movement."""
        if sash_index >= len(self.frames) - 1:
            return
            
        frame1 = self.frames[sash_index]
        frame2 = self.frames[sash_index + 1]
        
        # Calculate new sizes
        new_size1 = max(frame1['minsize'], frame1['current_size'] + delta)
        new_size2 = max(frame2['minsize'], frame2['current_size'] - delta)
        
        # Adjust if one frame hits minimum
        if new_size1 == frame1['minsize'] and delta < 0:
            delta = frame1['minsize'] - frame1['current_size']
            new_size2 = frame2['current_size'] - delta
        elif new_size2 == frame2['minsize'] and delta > 0:
            delta = frame2['current_size'] - frame2['minsize']
            new_size1 = frame1['current_size'] + delta
            new_size2 = frame2['minsize']
        
        # Update sizes
        frame1['current_size'] = new_size1
        frame2['current_size'] = new_size2
        
        self._layout_frames()
    
    def _layout_frames(self):
        """Layout frames and sashes."""
        current_pos = 0
        
        for i, frame_info in enumerate(self.frames):
            frame = frame_info['frame']
            size = frame_info['current_size']
            
            if self.orientation == 'vertical':
                frame.place(x=0, y=current_pos, relwidth=1, height=size)
                current_pos += size
                
                # Place sash if not the last frame
                if i < len(self.sashes):
                    sash = self.sashes[i]
                    sash.place(x=0, y=current_pos, relwidth=1, height=self.sash_size)
                    current_pos += self.sash_size
            else:
                frame.place(x=current_pos, y=0, width=size, relheight=1)
                current_pos += size
                
                # Place sash if not the last frame
                if i < len(self.sashes):
                    sash = self.sashes[i]
                    sash.place(x=current_pos, y=0, width=self.sash_size, relheight=1)
                    current_pos += self.sash_size
    
    def _apply_base_theme(self):
        """Apply base theme to the frame itself."""
        if self.theme:
            try:
                self.configure(
                    bg=self.theme.get("background", "#181A1B"),
                    highlightbackground=self.theme.get("foreground", "#F0F0F0")
                )
            except tk.TclError:
                pass
    
    def apply_theme(self, theme=None):
        """Apply theme to this frame and all its children recursively."""
        if theme:
            self.theme = theme
            
        # Apply theme to self
        self._apply_base_theme()
        
        # Apply theme to sashes
        self._apply_theme_to_sashes(self.theme)
        
        # Apply theme to all child widgets recursively
        for child in self.winfo_children():
            self._apply_theme_to_widget(child)
    
    def _apply_theme_to_widget(self, widget):
        """Apply theme to any widget recursively."""
        try:
            widget_class = widget.winfo_class()
            
            # Apply theming based on widget type
            if widget_class in ['Frame', 'LabelFrame']:
                widget.configure(
                    bg=self.theme.get("background", "#181A1B"),
                    fg=self.theme.get("foreground", "#F0F0F0")
                )
                if widget_class == 'LabelFrame':
                    widget.configure(
                        highlightbackground=self.theme.get("foreground", "#F0F0F0")
                    )
            elif widget_class == 'Button':
                btn_theme = self.theme.get("buttons", {}).get("DefaultBotton", {})
                widget.configure(
                    bg=btn_theme.get("background", "lightgray"),
                    fg=self.theme.get("foreground", "#F0F0F0"),
                    activebackground=btn_theme.get("active_background", self.theme.get("selection_color", "#00FF99")),
                    activeforeground=self.theme.get("foreground", "#F0F0F0")
                )
            elif widget_class == 'Label':
                widget.configure(
                    bg=self.theme.get("background", "#181A1B"),
                    fg=self.theme.get("foreground", "#F0F0F0")
                )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme.get("background_accent_color", "#23272A"),
                    fg=self.theme.get("foreground", "#F0F0F0"),
                    insertbackground=self.theme.get("foreground", "#F0F0F0"),
                    selectbackground=self.theme.get("selection_color", "#00FF99"),
                    selectforeground=self.theme.get("background", "#181A1B")
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme.get("data_field_background", "#23272A"),
                    fg=self.theme.get("foreground", "#F0F0F0"),
                    insertbackground=self.theme.get("foreground", "#F0F0F0"),
                    selectbackground=self.theme.get("selection_color", "#00FF99"),
                    selectforeground=self.theme.get("background", "#181A1B")
                )
            elif widget_class == 'Listbox':
                widget.configure(
                    bg=self.theme.get("background_accent_color", "#23272A"),
                    fg=self.theme.get("foreground", "#F0F0F0"),
                    selectbackground=self.theme.get("selection_color", "#00FF99"),
                    selectforeground=self.theme.get("background", "#181A1B")
                )
            elif widget_class == 'Checkbutton':
                widget.configure(
                    bg=self.theme.get("background", "#181A1B"),
                    fg=self.theme.get("foreground", "#F0F0F0"),
                    selectcolor=self.theme.get("background_accent_color", "#23272A"),
                    activebackground=self.theme.get("background", "#181A1B"),
                    activeforeground=self.theme.get("foreground", "#F0F0F0")
                )
            # Handle TTK widgets
            elif hasattr(widget, '__class__') and 'ttk' in str(widget.__class__):
                self._apply_ttk_theme(widget)
                
            # Recursively apply to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Widget doesn't support certain configurations, skip silently
            pass
    
    def _apply_ttk_theme(self, widget):
        """Apply theme to TTK widgets using styles."""
        try:
            style = ttk.Style()
            widget_class = widget.winfo_class()
            
            if widget_class == 'TCombobox':
                style.configure("Themed.TCombobox",
                              fieldbackground=self.theme.get("background_accent_color", "#23272A"),
                              background=self.theme.get("background", "#181A1B"),
                              foreground=self.theme.get("foreground", "#F0F0F0"),
                              selectbackground=self.theme.get("selection_color", "#00FF99"))
                widget.configure(style="Themed.TCombobox")
            elif widget_class == 'TScrollbar':
                style.configure("Themed.Vertical.TScrollbar",
                              background=self.theme.get("background_accent_color", "#23272A"),
                              troughcolor=self.theme.get("background", "#181A1B"),
                              bordercolor=self.theme.get("background_accent_color", "#23272A"),
                              arrowcolor=self.theme.get("foreground", "#F0F0F0"))
                widget.configure(style="Themed.Vertical.TScrollbar")
        except:
            # TTK styling can be complex, fail silently
            pass