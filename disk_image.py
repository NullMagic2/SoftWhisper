"""
PC-98 Disk Image Format Parsers
Supports: D88/D68, HDM, FDI, HDI, and raw sector images.
Provides uniform sector-level access regardless of container format.

Write support: each format implements write_sector() and save() so that
modifications can be flushed back to disk.
"""

import struct
import os
import shutil
import logging

log = logging.getLogger("pc98mount.disk")


class DiskImage:
    """Base class providing sector-level access to a disk image."""

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self._data = bytearray(f.read())
        self._parse()

    def _parse(self):
        raise NotImplementedError

    @property
    def sector_size(self):
        return self._sector_size

    @property
    def total_sectors(self):
        return self._total_sectors

    @property
    def label(self):
        return self._label

    def read_sector(self, lba):
        raise NotImplementedError

    def read_sectors(self, lba, count):
        data = bytearray()
        for i in range(count):
            data.extend(self.read_sector(lba + i))
        return bytes(data)

    # ── Write support ────────────────────────────────────────────────

    def write_sector(self, lba, data):
        """Write one sector. *data* must be at least sector_size bytes."""
        raise NotImplementedError

    def save(self, path=None):
        """Flush the in-memory image to disk.

        If *path* is given the image is written there (the original is
        untouched).  Otherwise the original file is overwritten via an
        atomic write-to-temp-then-rename pattern.
        """
        save_path = path or self.path
        if save_path == self.path:
            # Atomic overwrite: write next to the original, then rename.
            tmp = save_path + '.tmp'
            try:
                with open(tmp, 'wb') as f:
                    f.write(self._data)
                # On Windows, os.replace is atomic if on the same volume.
                shutil.move(tmp, save_path)
            except Exception:
                # Clean up partial write.
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        else:
            with open(save_path, 'wb') as f:
                f.write(self._data)
        log.info(f"Saved image ({len(self._data):,} bytes) to {save_path}")


class RawImage(DiskImage):
    """
    Raw / HDM image — flat sector dump with no header.
    PC-98 2HD: 77 cyl × 2 heads × 8 spt × 1024 bytes/sector = 1,261,568 bytes
    PC-98 2DD: 80 cyl × 2 heads × 9 spt × 512 bytes/sector  = 737,280 bytes
    """

    KNOWN_GEOMETRIES = {
        1261568: (1024, 1232),   # 2HD 1.2MB
        1228800: (1024, 1200),   # 2HD alternate
        737280:  (512, 1440),    # 2DD 720KB
        1474560: (512, 2880),    # 1.44MB (rare on PC-98)
    }

    def __init__(self, path, sector_size=None):
        self._forced_sector_size = sector_size
        super().__init__(path)

    def _parse(self):
        size = len(self._data)
        if self._forced_sector_size:
            self._sector_size = self._forced_sector_size
            self._total_sectors = size // self._sector_size
        elif size in self.KNOWN_GEOMETRIES:
            self._sector_size, self._total_sectors = self.KNOWN_GEOMETRIES[size]
        else:
            # Default: try 1024, fall back to 512
            if size % 1024 == 0:
                self._sector_size = 1024
            else:
                self._sector_size = 512
            self._total_sectors = size // self._sector_size
        self._label = f"RAW ({self._total_sectors} sectors)"

    def read_sector(self, lba):
        offset = lba * self._sector_size
        if offset + self._sector_size > len(self._data):
            return b'\x00' * self._sector_size
        return bytes(self._data[offset:offset + self._sector_size])

    def write_sector(self, lba, data):
        offset = lba * self._sector_size
        end = offset + self._sector_size
        if end > len(self._data):
            raise IndexError(f"Sector {lba} out of range")
        self._data[offset:end] = data[:self._sector_size]


class FDIImage(DiskImage):
    """
    FDI image format — 4096-byte header followed by raw sector data.
    Header contains geometry information.
    """

    HEADER_SIZE = 4096

    def _parse(self):
        if len(self._data) < self.HEADER_SIZE:
            raise ValueError("File too small for FDI format")

        fdd_type, hdr_size, sec_size = struct.unpack_from('<III', self._data, 0)
        spt, heads, cyls = struct.unpack_from('<III', self._data, 0x10)

        if sec_size not in (128, 256, 512, 1024, 2048, 4096):
            sec_size = 1024
        self._sector_size = sec_size
        self._total_sectors = (len(self._data) - self.HEADER_SIZE) // self._sector_size
        self._raw_offset = self.HEADER_SIZE
        self._label = f"FDI ({cyls}C/{heads}H/{spt}S)"

    def read_sector(self, lba):
        offset = self._raw_offset + lba * self._sector_size
        if offset + self._sector_size > len(self._data):
            return b'\x00' * self._sector_size
        return bytes(self._data[offset:offset + self._sector_size])

    def write_sector(self, lba, data):
        offset = self._raw_offset + lba * self._sector_size
        end = offset + self._sector_size
        if end > len(self._data):
            raise IndexError(f"Sector {lba} out of range")
        self._data[offset:end] = data[:self._sector_size]


class D88Image(DiskImage):
    """
    D88/D68 image format — used by many Japanese emulators.
    Has a header with disk name, media type, and a 164-entry track offset table.
    Each track's sectors have individual headers with C/H/R/N fields.
    """

    def _parse(self):
        if len(self._data) < 0x2B0:
            raise ValueError("File too small for D88 format")

        # D88 header
        name_raw = self._data[0:17].split(b'\x00')[0]
        try:
            self._label = name_raw.decode('shift_jis', errors='replace')
        except Exception:
            self._label = "D88 Image"

        self._write_protect = self._data[0x1A]
        self._media_type = self._data[0x1B]
        self._disk_size = struct.unpack_from('<I', self._data, 0x1C)[0]

        # Track offset table: 164 entries at offset 0x20
        self._track_offsets = []
        for i in range(164):
            off = struct.unpack_from('<I', self._data, 0x20 + i * 4)[0]
            self._track_offsets.append(off)

        # Build a flat sector table by walking all tracks
        self._sectors = []  # list of (offset_in_file, size)
        self._sector_size = 0

        for track_off in self._track_offsets:
            if track_off == 0:
                continue
            pos = track_off
            if pos >= len(self._data):
                continue

            while pos < len(self._data) - 16:
                c, h, r, n = struct.unpack_from('BBBB', self._data, pos)
                num_sects = struct.unpack_from('<H', self._data, pos + 4)[0]
                data_size = struct.unpack_from('<H', self._data, pos + 14)[0]

                if data_size == 0 or num_sects == 0:
                    break

                sec_size = 128 << n
                if self._sector_size == 0:
                    self._sector_size = sec_size

                data_offset = pos + 16
                self._sectors.append((data_offset, data_size))

                pos = data_offset + data_size

                if len(self._sectors) % num_sects == 0 and num_sects > 0:
                    if pos + 16 <= len(self._data):
                        next_c = self._data[pos]
                        next_h = self._data[pos + 1]
                        if next_c != c or next_h != h:
                            break

        if self._sector_size == 0:
            self._sector_size = 1024
        self._total_sectors = len(self._sectors)
        if not self._label or self._label.strip() == '':
            self._label = f"D88 ({self._total_sectors} sectors)"

    def read_sector(self, lba):
        if lba < 0 or lba >= len(self._sectors):
            return b'\x00' * self._sector_size
        offset, size = self._sectors[lba]
        data = self._data[offset:offset + size]
        if len(data) < self._sector_size:
            data = data + b'\x00' * (self._sector_size - len(data))
        return bytes(data[:self._sector_size])

    def write_sector(self, lba, data):
        if lba < 0 or lba >= len(self._sectors):
            raise IndexError(f"Sector {lba} out of range")
        offset, size = self._sectors[lba]
        # Write into the data portion, padded/truncated to stored size.
        write_data = bytearray(data[:size])
        if len(write_data) < size:
            write_data.extend(b'\x00' * (size - len(write_data)))
        self._data[offset:offset + size] = write_data


class HDIImage(DiskImage):
    """
    HDI image format — hard disk image with a small header.
    Used by Anex86 and some other emulators.

    Two layout variants exist:
      T98-Next / Neko Project II:
        0x00 – reserved,  0x04 – hdr_size,  0x08 – data_size,
        0x0C – sec_size,  0x10 – spt,  0x14 – heads,  0x18 – cyls
      Anex86 (fields shifted +4):
        0x00 – reserved,  0x04 – hdd_type,  0x08 – hdr_size,
        0x0C – data_size, 0x10 – sec_size,  0x14 – spt,
        0x18 – heads,     0x1C – cyls
    """

    _VALID_SECTOR_SIZES = (128, 256, 512, 1024, 2048, 4096)

    def _parse(self):
        if len(self._data) < 4096:
            raise ValueError("File too small for HDI format")

        # Try T98-style layout first (hdr_size at 0x04).
        hdr_size, sec_size, spt, heads, cyls = self._try_layout(0x04)
        if hdr_size is None:
            # Fall back to Anex86-style layout (hdr_size at 0x08).
            hdr_size, sec_size, spt, heads, cyls = self._try_layout(0x08)

        if hdr_size is None:
            # Neither layout produced sane values — use safe defaults.
            hdr_size = 4096
            sec_size = 512
            raw_len = len(self._data) - hdr_size
            spt = heads = cyls = 0
        else:
            raw_len = len(self._data) - hdr_size

        self._sector_size = sec_size
        self._raw_offset = hdr_size
        self._total_sectors = raw_len // sec_size
        self._spt = spt
        self._heads = heads
        if spt and heads and cyls:
            self._label = f"HDI ({cyls}C/{heads}H/{spt}S)"
        else:
            self._label = f"HDI ({self._total_sectors} sectors)"

    def _try_layout(self, hdr_offset):
        """Try reading HDI header fields starting at *hdr_offset*.
        Returns ``(hdr_size, sec_size, spt, heads, cyls)`` or
        ``(None, …)`` if the values don't look valid.
        """
        h = struct.unpack_from('<I', self._data, hdr_offset)[0]
        # data_size at hdr_offset+4 is informational; skip it.
        s = struct.unpack_from('<I', self._data, hdr_offset + 8)[0]
        spt = struct.unpack_from('<I', self._data, hdr_offset + 12)[0]
        heads = struct.unpack_from('<I', self._data, hdr_offset + 16)[0]
        cyls = struct.unpack_from('<I', self._data, hdr_offset + 20)[0]

        if h == 0 or h > 0x10000:
            return (None, None, None, None, None)
        if s not in self._VALID_SECTOR_SIZES:
            return (None, None, None, None, None)
        if spt == 0 or spt > 255 or heads == 0 or heads > 255:
            return (None, None, None, None, None)
        if cyls == 0 or cyls > 0xFFFF:
            return (None, None, None, None, None)
        return (h, s, spt, heads, cyls)

    def read_sector(self, lba):
        offset = self._raw_offset + lba * self._sector_size
        if offset + self._sector_size > len(self._data):
            return b'\x00' * self._sector_size
        return bytes(self._data[offset:offset + self._sector_size])

    def write_sector(self, lba, data):
        offset = self._raw_offset + lba * self._sector_size
        end = offset + self._sector_size
        if end > len(self._data):
            raise IndexError(f"Sector {lba} out of range")
        self._data[offset:end] = data[:self._sector_size]


def open_image(path):
    """Auto-detect image format and return appropriate DiskImage instance."""
    ext = path.lower()

    if ext.endswith('.d88') or ext.endswith('.d68') or ext.endswith('.d77'):
        return D88Image(path)
    elif ext.endswith('.fdi'):
        return FDIImage(path)
    elif ext.endswith('.hdi'):
        return HDIImage(path)
    elif ext.endswith('.hdm') or ext.endswith('.tfd'):
        return RawImage(path, sector_size=1024)
    elif ext.endswith('.img') or ext.endswith('.ima'):
        return RawImage(path)
    else:
        return RawImage(path)


# =============================================================================
# Blank image creation
# =============================================================================

# Pre-defined geometries: (cyls, heads, spt, sector_size)
BLANK_GEOMETRIES = {
    # ── Floppy ────────────────────────────────────────────────────
    "PC-98 2HD (1.2 MB)":   (77,  2,  8, 1024),
    "PC-98 2DD (640 KB)":   (80,  2,  8,  512),
    "PC-98 2DD (720 KB)":   (80,  2,  9,  512),
    "PC-98 1.44 MB":        (80,  2, 18,  512),
    # ── Hard Disk ─────────────────────────────────────────────────
    "HDD 20 MB":            (615,  4, 17, 512),
    "HDD 40 MB":            (615,  8, 17, 512),
    "HDD 80 MB":            (823,  8, 25, 512),
    "HDD 128 MB":           (1024, 8, 32, 512),
    "HDD 256 MB":           (1024, 8, 64, 512),
    "HDD 512 MB":           (1024,16, 64, 512),
    # ── Sentinel for custom geometry ──────────────────────────────
    "Custom":               None,
}

# Formats that the creator supports, with their default extension
BLANK_FORMATS = ["HDM", "D88", "FDI", "HDI", "RAW (.img)"]


def _build_fat_boot_sector(sector_size, spc, reserved, num_fats,
                           root_entries, fat_sectors, total_sectors,
                           media, spt, heads):
    """Build a minimal FAT12/16 boot sector (BPB) for a blank image."""
    boot = bytearray(sector_size)
    boot[0:3] = b'\xEB\x3C\x90'                     # JMP short + NOP
    boot[3:11] = b'PC98MTBL'                         # OEM name
    struct.pack_into('<H', boot, 0x0B, sector_size)  # bytes per sector
    boot[0x0D] = spc                                 # sectors per cluster
    struct.pack_into('<H', boot, 0x0E, reserved)     # reserved sectors
    boot[0x10] = num_fats                            # number of FATs
    struct.pack_into('<H', boot, 0x11, root_entries)
    struct.pack_into('<H', boot, 0x13,
                     total_sectors if total_sectors < 0x10000 else 0)
    boot[0x15] = media                               # media descriptor
    struct.pack_into('<H', boot, 0x16, fat_sectors)
    struct.pack_into('<H', boot, 0x18, spt)
    struct.pack_into('<H', boot, 0x1A, heads)
    struct.pack_into('<H', boot, 0x1C, 0)            # hidden sectors
    if total_sectors >= 0x10000:
        struct.pack_into('<I', boot, 0x20, total_sectors)
    return bytes(boot)


def _build_empty_fat(fat_type, fat_sectors, sector_size, media):
    """Build a zeroed FAT table with only the media-descriptor entries."""
    buf = bytearray(fat_sectors * sector_size)
    if fat_type == 12:
        # Entry 0 = 0xF00 | media, Entry 1 = 0xFFF
        buf[0] = media
        buf[1] = 0xFF
        buf[2] = 0xFF
    else:
        struct.pack_into('<H', buf, 0, 0xFF00 | media)
        struct.pack_into('<H', buf, 2, 0xFFFF)
    return bytes(buf)


def create_blank_image(path, fmt, geometry_name_or_tuple, format_fat=True):
    """Create a blank disk image at *path*.

    *fmt*                    – one of BLANK_FORMATS (e.g. "HDM", "D88", …)
    *geometry_name_or_tuple* – either a key into BLANK_GEOMETRIES or a raw
                               ``(cyls, heads, spt, sector_size)`` tuple.
    *format_fat*             – if True, write a valid FAT12/16 boot sector
                               and empty FAT so the image is ready to use.

    Returns the opened DiskImage instance.
    """
    if isinstance(geometry_name_or_tuple, str):
        geom = BLANK_GEOMETRIES[geometry_name_or_tuple]
        if geom is None:
            raise ValueError(
                "The 'Custom' geometry requires a (cyls, heads, spt, "
                "sector_size) tuple — not the string 'Custom'.")
        cyls, heads, spt, sector_size = geom
    else:
        cyls, heads, spt, sector_size = geometry_name_or_tuple

    total_sectors = cyls * heads * spt
    image_bytes = total_sectors * sector_size

    if image_bytes == 0:
        raise ValueError("Image size would be 0 bytes.")

    # --- Compute FAT parameters for any volume size ---------------
    fat_params = _compute_fat_params(total_sectors, sector_size,
                                     image_bytes, spt, heads)

    spc          = fat_params['spc']
    reserved     = fat_params['reserved']
    num_fats     = fat_params['num_fats']
    root_entries = fat_params['root_entries']
    fat_sectors  = fat_params['fat_sectors']
    media        = fat_params['media']
    fat_type     = fat_params['fat_type']

    boot_sector = (_build_fat_boot_sector(
        sector_size, spc, reserved, num_fats,
        root_entries, fat_sectors, total_sectors, media,
        spt, heads) if format_fat else b'\x00' * sector_size)

    fat_data = (_build_empty_fat(
        fat_type, fat_sectors, sector_size, media
    ) if format_fat else b'\x00' * (fat_sectors * sector_size))

    # ── Build the raw flat image ──────────────────────────────────
    raw = bytearray(image_bytes)
    raw[0:sector_size] = boot_sector[:sector_size]

    if format_fat:
        fat_off = reserved * sector_size
        for i in range(num_fats):
            off = fat_off + i * fat_sectors * sector_size
            raw[off:off + len(fat_data)] = fat_data

    # ── Write to the requested container format ───────────────────
    fmt_up = fmt.upper()
    if fmt_up == "D88":
        _write_d88(path, raw, cyls, heads, spt, sector_size)
    elif fmt_up == "FDI":
        _write_fdi(path, raw, cyls, heads, spt, sector_size)
    elif fmt_up == "HDI":
        _write_hdi(path, raw, cyls, heads, spt, sector_size)
    else:
        # HDM or RAW — flat dump
        with open(path, 'wb') as f:
            f.write(raw)

    log.info(f"Created blank {fmt} image: {path} "
             f"({cyls}C/{heads}H/{spt}S, {sector_size}B, "
             f"{image_bytes:,} bytes, FAT{fat_type if format_fat else 'none'})")

    return open_image(path)


def _compute_fat_params(total_sectors, sector_size, image_bytes, spt, heads):
    """Choose FAT12 or FAT16 and compute all BPB/FAT layout parameters.

    Works for everything from a 640 KB floppy to a 2 GB hard disk.
    """
    num_fats = 2
    reserved = 1

    # ── Pick media descriptor ─────────────────────────────────────
    if image_bytes <= 1_474_560:
        # Floppy-class
        if sector_size == 1024:
            media = 0xFE          # PC-98 2HD
        elif image_bytes <= 737_280:
            media = 0xFD          # 2DD
        else:
            media = 0xF0          # 1.44 MB / generic
    else:
        media = 0xF8              # hard disk

    # ── Root directory entries ────────────────────────────────────
    if image_bytes <= 1_474_560:
        root_entries = 192 if sector_size == 1024 else (
            112 if image_bytes <= 737_280 else 224)
    else:
        root_entries = 512        # typical for HDD

    root_dir_sectors = (root_entries * 32 + sector_size - 1) // sector_size

    # ── Sectors-per-cluster (SPC) and FAT type ────────────────────
    # PC-98 floppies typically use SPC=1 for 1024-byte sectors and
    # SPC=2 for small 512-byte floppies.  For hard disks we pick
    # SPC so that each cluster is a reasonable size and the total
    # cluster count lands in the right FAT12/16 range.
    if image_bytes <= 1_474_560:
        # Floppy: keep it simple
        if sector_size == 1024:
            spc = 1
        elif image_bytes <= 737_280:
            spc = 2
        else:
            spc = 1
    else:
        # Hard disk — choose SPC to stay within FAT16 limits.
        # FAT16 supports up to 65,524 clusters.  We also want
        # clusters large enough to keep the FAT table reasonable.
        # Standard DOS cluster sizes by volume:
        #   <=  128 MB  →  2 KB clusters  (SPC=4  for 512B sectors)
        #   <=  256 MB  →  4 KB clusters  (SPC=8)
        #   <=  512 MB  →  8 KB clusters  (SPC=16)
        #   <= 1024 MB  → 16 KB clusters  (SPC=32)
        #   <= 2048 MB  → 32 KB clusters  (SPC=64)
        mb = image_bytes / (1024 * 1024)
        if mb <= 128:
            spc = 4
        elif mb <= 256:
            spc = 8
        elif mb <= 512:
            spc = 16
        elif mb <= 1024:
            spc = 32
        else:
            spc = 64

    # ── Determine FAT type from cluster count ─────────────────────
    data_sectors = total_sectors - reserved - num_fats * 1 - root_dir_sectors
    est_clusters = data_sectors // spc
    fat_type = 12 if est_clusters < 4085 else 16

    # ── Compute exact FAT size (iterative) ────────────────────────
    # The FAT itself consumes sectors, which reduces the data area,
    # which changes the cluster count, which changes the FAT size.
    # We iterate until stable.
    if fat_type == 12:
        bytes_per_fat_entry = 1.5   # 12 bits
    else:
        bytes_per_fat_entry = 2     # 16 bits

    fat_sectors = 1
    for _ in range(20):
        data_sects = (total_sectors - reserved
                      - num_fats * fat_sectors - root_dir_sectors)
        if data_sects <= 0:
            break
        clusters = data_sects // spc
        needed_bytes = int((clusters + 2) * bytes_per_fat_entry + 0.5)
        needed_sects = (needed_bytes + sector_size - 1) // sector_size
        if needed_sects <= fat_sectors:
            break
        fat_sectors = needed_sects

    return {
        'spc':          spc,
        'reserved':     reserved,
        'num_fats':     num_fats,
        'root_entries': root_entries,
        'fat_sectors':  fat_sectors,
        'media':        media,
        'fat_type':     fat_type,
    }


# ── Container writers ────────────────────────────────────────────

def _write_d88(path, raw, cyls, heads, spt, sector_size):
    """Wrap flat *raw* data in a D88 container and write to *path*."""
    n_val = {128: 0, 256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5}.get(
        sector_size, 3)
    num_tracks = cyls * heads

    # Pre-calculate track offsets (header = 0x2B0 bytes)
    header_size = 0x2B0
    track_offsets = []
    current_offset = header_size
    for _ in range(num_tracks):
        track_offsets.append(current_offset)
        # Each sector has a 16-byte header + sector_size data
        current_offset += spt * (16 + sector_size)
    # Pad the offset table to 164 entries
    while len(track_offsets) < 164:
        track_offsets.append(0)

    disk_size = current_offset

    # Build header
    hdr = bytearray(header_size)
    hdr[0:16] = b'BLANK\x00' + b'\x00' * 10  # disk name
    hdr[0x1A] = 0x00        # write protect: off
    hdr[0x1B] = 0x00        # media type: 2D (generic)
    struct.pack_into('<I', hdr, 0x1C, disk_size)
    for i, off in enumerate(track_offsets):
        struct.pack_into('<I', hdr, 0x20 + i * 4, off)

    # Build track/sector data
    body = bytearray()
    raw_pos = 0
    for trk in range(num_tracks):
        c = trk // heads
        h = trk % heads
        for s in range(spt):
            # 16-byte sector header
            sec_hdr = bytearray(16)
            sec_hdr[0] = c       # C
            sec_hdr[1] = h       # H
            sec_hdr[2] = s + 1   # R (1-based)
            sec_hdr[3] = n_val   # N
            struct.pack_into('<H', sec_hdr, 4, spt)         # sectors in track
            sec_hdr[6] = 0       # density: MFM
            sec_hdr[7] = 0       # deleted mark
            sec_hdr[8] = 0       # status
            # bytes 9-13: reserved
            struct.pack_into('<H', sec_hdr, 14, sector_size) # data size

            body.extend(sec_hdr)
            body.extend(raw[raw_pos:raw_pos + sector_size])
            raw_pos += sector_size

    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(body)


def _write_fdi(path, raw, cyls, heads, spt, sector_size):
    """Wrap flat *raw* data in an FDI container and write to *path*."""
    hdr = bytearray(FDIImage.HEADER_SIZE)
    struct.pack_into('<I', hdr, 0x00, 0)           # fdd_type
    struct.pack_into('<I', hdr, 0x04, FDIImage.HEADER_SIZE)
    struct.pack_into('<I', hdr, 0x08, sector_size)  # sector size
    struct.pack_into('<I', hdr, 0x10, spt)
    struct.pack_into('<I', hdr, 0x14, heads)
    struct.pack_into('<I', hdr, 0x18, cyls)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(raw)


def _write_hdi(path, raw, cyls, heads, spt, sector_size):
    """Wrap flat *raw* data in an HDI container and write to *path*."""
    hdr_size = 4096
    hdr = bytearray(hdr_size)
    struct.pack_into('<I', hdr, 0x04, hdr_size)
    struct.pack_into('<I', hdr, 0x08, len(raw))
    struct.pack_into('<I', hdr, 0x0C, sector_size)
    struct.pack_into('<I', hdr, 0x10, spt)
    struct.pack_into('<I', hdr, 0x14, heads)
    struct.pack_into('<I', hdr, 0x18, cyls)
    with open(path, 'wb') as f:
        f.write(hdr)
        f.write(raw)