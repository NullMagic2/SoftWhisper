# PC-98 Disk Image Mounter

Mount NEC PC-98 floppy and hard disk images as native Windows drive letters.
Browse in Explorer, inspect sectors in a hex viewer, or export raw data.

**Zero external dependencies** — just Python 3.8+ with tkinter.

## How It Works

Mounting uses Windows' built-in `subst` command: image contents are extracted
to a temp directory, then mapped to a drive letter. When you unmount, the
temp directory is cleaned up. The original image is never modified.

## Supported Formats

| Extension       | Format                              |
|-----------------|-------------------------------------|
| `.d88` `.d68`   | D88 (common in Japanese emulators)  |
| `.hdm` `.tfd`   | Raw 2HD (1.2 MB, 1024 B sectors)   |
| `.fdi`          | FDI (4096-byte header + raw data)   |
| `.hdi`          | HDI (Anex86 hard disk image)        |
| `.img` `.ima`   | Raw sector dump (auto-detected)     |

## Usage

### GUI

```
python pc98mount.py
```

1. Click **Add Image…** to load disk images
2. Select a **Mode**:
   - `fat` — standard filesystem view (files and folders)
   - `flat` — raw image as a single `DISK.IMG` file (for hex editors, IDA)
   - `sectors` — each sector as `SECTOR_NNNN.BIN` (for scripting)
3. Pick a drive letter, click **Mount**
4. Click **Open in Explorer** or just navigate to the drive letter
5. **Unmount** when done

### Command line

```bash
python pc98mount.py game.d88                      # open in GUI
python pc98mount.py game.d88 -d P                 # auto-mount FAT at P:
python pc98mount.py game.d88 -d P -m flat         # auto-mount raw at P:
python pc98mount.py game.d88 -d Q -m sectors      # sectors at Q:
```

### Hex Viewer (built-in)

The **Hex Viewer** tab provides sector-level inspection:

- Navigate sectors by number (decimal or `0x` hex)
- Offsets shown relative to sector or absolute in image
- Boot sector BPB fields auto-decoded with names and values
- x86 signature detection (`INT 21h`, `INT 1Bh`, boot markers)
- Hex pattern search across entire image (e.g., `CD 1B`)
- Bookmark interesting sectors
- Export arbitrary sector ranges to `.bin` files
- Color coding: zero bytes dimmed, high bytes highlighted, ASCII in green

### Working with non-FAT images

Many PC-98 games bypass FAT and use raw sector I/O. For these:

1. Mount in **flat** mode → open `P:\DISK.IMG` in HxD or your disassembler
2. Mount in **sectors** mode → script batch analysis:
   ```python
   for f in os.listdir("Q:\\"):
       with open(f"Q:\\{f}", "rb") as fh:
           data = fh.read()
           # scan for patterns, extract graphics, etc.
   ```
3. Use the **Hex Viewer** to search for code patterns (`CD 1B` = INT 1Bh)
4. Export sector ranges for targeted analysis in Ghidra/IDA

## Project Structure

```
pc98mount/
├── pc98mount.py       # GUI and entry point
├── disk_image.py      # Image format parsers (D88, HDM, FDI, HDI, raw)
├── fat_fs.py          # FAT12/FAT16 filesystem parser
├── hex_viewer.py      # Sector-level hex viewer widget
├── mount_backend.py   # Mount strategies (subst, WinFsp detection)
├── requirements.txt   # (empty — no external deps)
└── README.md
```

## Troubleshooting

**Mount fails** — You may need to run as Administrator for certain drive letters.

**No files shown in Files tab** — The disk has no FAT filesystem.
Use flat/sector mode or the hex viewer.

**Shift-JIS filenames garbled** — Expected if your locale isn't Japanese.
The underlying data is correct; extraction works fine.

**Large images slow in sectors mode** — Each sector becomes a file in the
temp dir. For multi-megabyte HDD images, flat mode is faster.

## License

Public domain / MIT.
