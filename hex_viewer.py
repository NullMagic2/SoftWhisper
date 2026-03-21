"""
Sector-level hex viewer widget for PC-98 disk images (wxPython).
Provides a traditional hex editor view with sector navigation,
bookmarking, search, and raw export.
"""

import wx
import struct


# Known PC-98 I/O and structure signatures for auto-annotation
KNOWN_SIGNATURES = {
    0: "Boot Sector / IPL",
}

# Common byte patterns to flag
BYTE_PATTERNS = {
    b'\xEB': 'JMP short (x86)',
    b'\xE9': 'JMP near (x86)',
    b'\xCD\x1B': 'INT 1Bh (PC-98 BIOS disk)',
    b'\xCD\x21': 'INT 21h (DOS)',
    b'\xCD\x18': 'INT 18h (PC-98 BIOS)',
    b'\x55\xAA': 'Boot signature',
    b'\xEB\x3C\x90': 'DOS boot jump',
}

# PC-98 specific structures at known BPB offsets
BPB_FIELDS = {
    0x0B: ("Bytes/Sector", "<H"),
    0x0D: ("Sects/Cluster", "B"),
    0x0E: ("Reserved Sects", "<H"),
    0x10: ("Num FATs", "B"),
    0x11: ("Root Entries", "<H"),
    0x13: ("Total Sects 16", "<H"),
    0x15: ("Media Desc", "B"),
    0x16: ("FAT Size", "<H"),
    0x18: ("Sects/Track", "<H"),
    0x1A: ("Num Heads", "<H"),
    0x1C: ("Hidden Sects", "<H"),
}


class HexViewerPanel(wx.Panel):
    """
    A sector-level hex viewer with navigation and analysis features.
    Embeds into a parent wx container.
    """

    BYTES_PER_ROW = 16

    def __init__(self, parent):
        super().__init__(parent)
        self.disk = None
        self.current_sector = 0
        self.bookmarks = {}
        self.sector_annotations = {}
        self._search_bytes = None
        self._search_pos = 0
        self._offset_absolute = False
        self._build_ui()

    def set_disk(self, disk_image):
        """Attach a disk image to this viewer."""
        self.disk = disk_image
        self.current_sector = 0
        self.bookmarks.clear()
        self.sector_annotations.clear()
        self._update_nav_limits()
        self._show_sector(0)

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # --- Navigation bar ---
        nav_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_first = wx.Button(self, label="\u25C0\u25C0", size=(36, -1))
        btn_prev = wx.Button(self, label="\u25C0", size=(36, -1))
        btn_first.Bind(wx.EVT_BUTTON, lambda e: self._go_first())
        btn_prev.Bind(wx.EVT_BUTTON, lambda e: self._go_prev())
        nav_sizer.Add(btn_first, 0, wx.RIGHT, 1)
        nav_sizer.Add(btn_prev, 0, wx.RIGHT, 4)

        nav_sizer.Add(wx.StaticText(self, label="Sector:"),
                       0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.sector_ctrl = wx.TextCtrl(self, value="0", size=(80, -1),
                                        style=wx.TE_PROCESS_ENTER)
        self.sector_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda e: self._go_to_sector())
        nav_sizer.Add(self.sector_ctrl, 0, wx.RIGHT, 2)
        btn_go = wx.Button(self, label="Go", size=(40, -1))
        btn_go.Bind(wx.EVT_BUTTON, lambda e: self._go_to_sector())
        nav_sizer.Add(btn_go, 0, wx.RIGHT, 4)

        btn_next = wx.Button(self, label="\u25B6", size=(36, -1))
        btn_last = wx.Button(self, label="\u25B6\u25B6", size=(36, -1))
        btn_next.Bind(wx.EVT_BUTTON, lambda e: self._go_next())
        btn_last.Bind(wx.EVT_BUTTON, lambda e: self._go_last())
        nav_sizer.Add(btn_next, 0, wx.RIGHT, 1)
        nav_sizer.Add(btn_last, 0, wx.RIGHT, 8)

        self.sector_label = wx.StaticText(self, label="/ 0")
        nav_sizer.Add(self.sector_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)

        nav_sizer.Add(wx.StaticText(self, label="Show:"),
                       0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.rb_sector = wx.RadioButton(self, label="Sector", style=wx.RB_GROUP)
        self.rb_absolute = wx.RadioButton(self, label="Absolute")
        self.rb_sector.Bind(wx.EVT_RADIOBUTTON, self._on_offset_mode)
        self.rb_absolute.Bind(wx.EVT_RADIOBUTTON, self._on_offset_mode)
        nav_sizer.Add(self.rb_sector, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        nav_sizer.Add(self.rb_absolute, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)

        btn_bm = wx.Button(self, label="Bookmark", size=(80, -1))
        btn_bm.Bind(wx.EVT_BUTTON, lambda e: self._add_bookmark())
        btn_bm_list = wx.Button(self, label="Bookmarks\u2026", size=(100, -1))
        btn_bm_list.Bind(wx.EVT_BUTTON, lambda e: self._show_bookmarks())
        nav_sizer.Add(btn_bm, 0, wx.RIGHT, 2)
        nav_sizer.Add(btn_bm_list, 0)
        main_sizer.Add(nav_sizer, 0, wx.EXPAND | wx.BOTTOM, 4)

        # --- Tools bar ---
        tools_sizer = wx.BoxSizer(wx.HORIZONTAL)
        tools_sizer.Add(wx.StaticText(self, label="Search hex:"),
                         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.search_ctrl = wx.TextCtrl(self, size=(200, -1),
                                        style=wx.TE_PROCESS_ENTER)
        self.search_ctrl.Bind(wx.EVT_TEXT_ENTER, lambda e: self._search_hex())
        tools_sizer.Add(self.search_ctrl, 0, wx.RIGHT, 2)
        btn_find = wx.Button(self, label="Find", size=(50, -1))
        btn_find.Bind(wx.EVT_BUTTON, lambda e: self._search_hex())
        btn_find_next = wx.Button(self, label="Find Next", size=(70, -1))
        btn_find_next.Bind(wx.EVT_BUTTON, lambda e: self._search_next())
        tools_sizer.Add(btn_find, 0, wx.RIGHT, 2)
        tools_sizer.Add(btn_find_next, 0, wx.RIGHT, 12)

        tools_sizer.Add(wx.StaticText(self, label="Export sectors:"),
                         0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 4)
        self.export_from_ctrl = wx.TextCtrl(self, value="0", size=(60, -1))
        tools_sizer.Add(self.export_from_ctrl, 0, wx.RIGHT, 2)
        tools_sizer.Add(wx.StaticText(self, label="to"),
                         0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 4)
        self.export_to_ctrl = wx.TextCtrl(self, value="0", size=(60, -1))
        tools_sizer.Add(self.export_to_ctrl, 0, wx.RIGHT, 4)
        btn_export = wx.Button(self, label="Export\u2026", size=(70, -1))
        btn_export.Bind(wx.EVT_BUTTON, lambda e: self._export_range())
        tools_sizer.Add(btn_export, 0)
        main_sizer.Add(tools_sizer, 0, wx.EXPAND | wx.BOTTOM, 4)

        # --- Hex display ---
        mono = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                        wx.FONTWEIGHT_NORMAL, faceName="Consolas")
        # Try to use Consolas; fall back to system monospace
        if not mono.IsOk() or mono.GetFaceName() != "Consolas":
            mono = wx.Font(10, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                            wx.FONTWEIGHT_NORMAL)

        self.hex_text = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
                  | wx.TE_DONTWRAP | wx.HSCROLL,
        )
        self.hex_text.SetFont(mono)
        self.hex_text.SetBackgroundColour(wx.Colour(30, 30, 46))
        self.hex_text.SetForegroundColour(wx.Colour(205, 214, 244))
        main_sizer.Add(self.hex_text, 1, wx.EXPAND | wx.BOTTOM, 4)

        # --- Annotation bar ---
        self.annot_label = wx.StaticText(self, label="")
        annot_font = wx.Font(9, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL,
                              wx.FONTWEIGHT_NORMAL)
        self.annot_label.SetFont(annot_font)
        main_sizer.Add(self.annot_label, 0, wx.EXPAND)

        self.SetSizer(main_sizer)

    # ── Colour helpers ───────────────────────────────────────────────

    # Catppuccin Mocha palette used by the original tkinter viewer
    _CLR_OFFSET  = wx.Colour(137, 180, 250)   # blue
    _CLR_BYTE    = wx.Colour(205, 214, 244)   # text
    _CLR_ZERO    = wx.Colour(88, 91, 112)     # dim
    _CLR_HIGH    = wx.Colour(249, 226, 175)   # yellow
    _CLR_ASCII   = wx.Colour(166, 227, 161)   # green
    _CLR_DOT     = wx.Colour(88, 91, 112)     # dim
    _CLR_SEP     = wx.Colour(69, 71, 90)
    _CLR_HEADER  = wx.Colour(137, 180, 250)
    _CLR_BPB     = wx.Colour(203, 166, 247)   # mauve
    _CLR_SIG     = wx.Colour(243, 139, 168)   # red
    _CLR_SEARCH  = wx.Colour(249, 226, 175)   # yellow bg
    _CLR_SRCHFG  = wx.Colour(30, 30, 46)      # dark fg for search hits

    def _append(self, text, colour=None, bold=False):
        """Append coloured text to the hex display."""
        start = self.hex_text.GetLastPosition()
        self.hex_text.AppendText(text)
        end = self.hex_text.GetLastPosition()
        if colour or bold:
            attr = wx.TextAttr()
            if colour:
                attr.SetTextColour(colour)
            if bold:
                f = self.hex_text.GetFont()
                f.SetWeight(wx.FONTWEIGHT_BOLD)
                attr.SetFont(f)
            self.hex_text.SetStyle(start, end, attr)

    # ── Navigation ───────────────────────────────────────────────────

    def _update_nav_limits(self):
        if self.disk:
            self.sector_label.SetLabel(f"/ {self.disk.total_sectors - 1}")
        else:
            self.sector_label.SetLabel("/ 0")

    def _go_first(self):
        self._show_sector(0)

    def _go_prev(self):
        if self.current_sector > 0:
            self._show_sector(self.current_sector - 1)

    def _go_next(self):
        if self.disk and self.current_sector < self.disk.total_sectors - 1:
            self._show_sector(self.current_sector + 1)

    def _go_last(self):
        if self.disk:
            self._show_sector(self.disk.total_sectors - 1)

    def _go_to_sector(self):
        try:
            s = self.sector_ctrl.GetValue().strip()
            if s.startswith('0x') or s.startswith('0X'):
                sector = int(s, 16)
            else:
                sector = int(s)
            if self.disk and 0 <= sector < self.disk.total_sectors:
                self._show_sector(sector)
            else:
                wx.MessageBox(
                    f"Sector must be 0\u2013"
                    f"{self.disk.total_sectors - 1 if self.disk else 0}",
                    "Invalid Sector", wx.OK | wx.ICON_WARNING)
        except ValueError:
            wx.MessageBox("Enter a decimal or hex (0x\u2026) sector number.",
                          "Invalid Input", wx.OK | wx.ICON_WARNING)

    def _on_offset_mode(self, event):
        self._offset_absolute = self.rb_absolute.GetValue()
        self._refresh()

    # ── Display ──────────────────────────────────────────────────────

    def _show_sector(self, sector_num):
        if not self.disk:
            return
        self.current_sector = sector_num
        self.sector_ctrl.SetValue(str(sector_num))
        data = self.disk.read_sector(sector_num)
        self._render_hex(data, sector_num)

    def _refresh(self):
        self._show_sector(self.current_sector)

    def _render_hex(self, data, sector_num):
        self.hex_text.SetEditable(True)
        self.hex_text.Clear()

        sector_size = len(data)
        abs_offset_base = sector_num * sector_size
        use_absolute = self._offset_absolute

        # Header
        header = f"{'Offset':>10s}  "
        for i in range(self.BYTES_PER_ROW):
            header += f"{i:02X} "
        header += " ASCII\n"
        self._append(header, self._CLR_HEADER, bold=True)
        self._append("\u2500" * 78 + "\n", self._CLR_SEP)

        # Annotations
        annot_parts = []
        if sector_num in KNOWN_SIGNATURES:
            annot_parts.append(KNOWN_SIGNATURES[sector_num])

        is_boot = (sector_num == 0 and len(data) >= 64
                    and data[0] in (0xEB, 0xE9))

        for row_start in range(0, sector_size, self.BYTES_PER_ROW):
            # Offset
            if use_absolute:
                off_val = abs_offset_base + row_start
                off_str = f"0x{off_val:08X}"
            else:
                off_str = f"0x{row_start:04X}"
            self._append(f"{off_str:>10s}  ", self._CLR_OFFSET)

            # Hex bytes
            row_data = data[row_start:row_start + self.BYTES_PER_ROW]
            for i, byte in enumerate(row_data):
                if is_boot and (row_start + i) in BPB_FIELDS:
                    clr = self._CLR_BPB
                elif byte == 0x00:
                    clr = self._CLR_ZERO
                elif byte >= 0x80:
                    clr = self._CLR_HIGH
                else:
                    clr = self._CLR_BYTE
                self._append(f"{byte:02X} ", clr)

            if len(row_data) < self.BYTES_PER_ROW:
                self._append("   " * (self.BYTES_PER_ROW - len(row_data)))

            self._append(" ", self._CLR_SEP)

            # ASCII
            for byte in row_data:
                if 0x20 <= byte <= 0x7E:
                    self._append(chr(byte), self._CLR_ASCII)
                else:
                    self._append("\u00B7", self._CLR_DOT)

            self._append("\n")

        # BPB fields for boot sector
        if is_boot:
            self._append("\n", None)
            self._append("\u2500\u2500\u2500 BPB Fields "
                          "\u2500" * 25 + "\n", self._CLR_HEADER, True)
            for off, (name, fmt) in sorted(BPB_FIELDS.items()):
                try:
                    if fmt == 'B':
                        val = data[off]
                        val_str = f"{val} (0x{val:02X})"
                    else:
                        val = struct.unpack_from(fmt, data, off)[0]
                        val_str = f"{val} (0x{val:04X})"
                    self._append(
                        f"  0x{off:04X}  {name:<18s} = {val_str}\n",
                        self._CLR_BPB)
                except (struct.error, IndexError):
                    pass

            sigs = self._find_signatures(data)
            if sigs:
                self._append("\n\u2500\u2500\u2500 Signatures "
                              "\u2500" * 25 + "\n", self._CLR_HEADER, True)
                for off, label in sigs:
                    self._append(f"  0x{off:04X}  {label}\n",
                                  self._CLR_SIG, True)

        # Annotation bar
        if sector_num in self.bookmarks:
            annot_parts.append(f"Bookmark: {self.bookmarks[sector_num]}")
        if annot_parts:
            self.annot_label.SetLabel("  \u2502  ".join(annot_parts))
        else:
            self.annot_label.SetLabel(
                f"Sector {sector_num} \u2014 {sector_size} bytes"
                f" \u2014 Abs offset 0x{abs_offset_base:X}")

        self.hex_text.SetInsertionPoint(0)
        self.hex_text.SetEditable(False)

    def _find_signatures(self, data):
        results = []
        for pattern, label in BYTE_PATTERNS.items():
            idx = 0
            while True:
                pos = data.find(pattern, idx)
                if pos == -1:
                    break
                results.append((pos, label))
                idx = pos + 1
        results.sort(key=lambda x: x[0])
        return results

    # ── Search ───────────────────────────────────────────────────────

    def _search_hex(self):
        query = self.search_ctrl.GetValue().strip()
        if not query or not self.disk:
            return
        try:
            cleaned = query.replace('0x', '').replace('0X', '') \
                           .replace(',', ' ')
            hex_bytes = bytes.fromhex(cleaned.replace(' ', ''))
        except ValueError:
            hex_bytes = query.encode('ascii', errors='replace')

        self._search_bytes = hex_bytes
        self._search_pos = self.current_sector * self.disk.sector_size
        self._do_search()

    def _search_next(self):
        if self._search_bytes and self.disk:
            self._search_pos += 1
            self._do_search()

    def _do_search(self):
        if not self._search_bytes or not self.disk:
            return
        pattern = self._search_bytes
        sector_size = self.disk.sector_size
        total = self.disk.total_sectors

        start_sector = self._search_pos // sector_size
        start_offset = self._search_pos % sector_size

        for s in range(start_sector, total):
            data = self.disk.read_sector(s)
            search_start = start_offset if s == start_sector else 0
            pos = data.find(pattern, search_start)
            if pos != -1:
                self._search_pos = s * sector_size + pos
                self._show_sector(s)
                self.annot_label.SetLabel(
                    f"Found at sector {s}, offset 0x{pos:X} "
                    f"(absolute 0x{s * sector_size + pos:X})")
                return

        wx.MessageBox("Pattern not found (searched to end of image).",
                      "Not Found", wx.OK | wx.ICON_INFORMATION)

    # ── Bookmarks ────────────────────────────────────────────────────

    def _add_bookmark(self):
        if not self.disk:
            return
        default = f"Sector {self.current_sector}"
        dlg = wx.TextEntryDialog(
            self, f"Label for sector {self.current_sector}:",
            "Add Bookmark", default)
        dlg.CentreOnParent()
        if dlg.ShowModal() == wx.ID_OK:
            self.bookmarks[self.current_sector] = dlg.GetValue()
            self._refresh()
        dlg.Destroy()

    def _show_bookmarks(self):
        if not self.bookmarks:
            wx.MessageBox("No bookmarks yet.\nUse the Bookmark button "
                          "to mark sectors.",
                          "Bookmarks", wx.OK | wx.ICON_INFORMATION)
            return

        sorted_bm = sorted(self.bookmarks.items())
        choices = [f"Sector {s:>6d}: {lbl}" for s, lbl in sorted_bm]

        dlg = wx.SingleChoiceDialog(
            self, "Select a bookmark to jump to:", "Bookmarks", choices)
        dlg.CentreOnParent()
        if dlg.ShowModal() == wx.ID_OK:
            idx = dlg.GetSelection()
            self._show_sector(sorted_bm[idx][0])
        dlg.Destroy()

    # ── Export ───────────────────────────────────────────────────────

    def _export_range(self):
        if not self.disk:
            return
        try:
            from_s = int(self.export_from_ctrl.GetValue())
            to_s = int(self.export_to_ctrl.GetValue())
        except ValueError:
            wx.MessageBox("Enter valid sector numbers.",
                          "Invalid Range", wx.OK | wx.ICON_WARNING)
            return

        if from_s < 0 or to_s >= self.disk.total_sectors or from_s > to_s:
            wx.MessageBox(
                f"Sectors must be 0\u2013{self.disk.total_sectors - 1}, "
                f"from \u2264 to.",
                "Invalid Range", wx.OK | wx.ICON_WARNING)
            return

        dlg = wx.FileDialog(
            self, "Export Sector Range",
            defaultFile=f"sectors_{from_s}-{to_s}.bin",
            wildcard="Binary (*.bin)|*.bin|All Files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        dlg.CentreOnParent()
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            data = self.disk.read_sectors(from_s, to_s - from_s + 1)
            with open(path, 'wb') as f:
                f.write(data)
            wx.MessageBox(
                f"Exported sectors {from_s}\u2013{to_s} "
                f"({len(data):,} bytes) to:\n{path}",
                "Exported", wx.OK | wx.ICON_INFORMATION)
        dlg.Destroy()