"""
FAT12/FAT16 Filesystem Parser for PC-98 disk images.

Key design: the BPB's bytes_per_sector may differ from the disk image's
native sector size (e.g., BPB says 1024 but a D88 image exposes 4096-byte
logical sectors). All filesystem reads go through _read_fs_bytes() which
translates BPB byte offsets into disk sector reads, making it work
regardless of how the disk image is sliced.

Write-back support: write_back_from_directory() rebuilds the FAT, root
directory, and data area from a real directory on the host filesystem,
then saves the result through the disk image's write_sector()/save() API.
"""

import os
import struct
import logging
from datetime import datetime

from partition import detect_partitions

log = logging.getLogger("pc98mount.fat")

# FAT constants
FAT12_EOC = 0xFF8
FAT16_EOC = 0xFFF8
FAT12_BAD = 0xFF7
FAT16_BAD = 0xFFF7

# Directory entry attributes
ATTR_READ_ONLY = 0x01
ATTR_HIDDEN    = 0x02
ATTR_SYSTEM    = 0x04
ATTR_VOLUME_ID = 0x08
ATTR_DIRECTORY = 0x10
ATTR_ARCHIVE   = 0x20
ATTR_LFN       = 0x0F

# Known PC-98 floppy geometries:
PC98_KNOWN_GEOMETRIES = [
    (1261568, 1024, 1, 1, 2, 192, 2, 1232, 0xFE),   # 2HD 1.2MB (most common)
    (1228800, 1024, 1, 1, 2, 192, 2, 1200, 0xFE),   # 2HD variant
    (655360,  512,  2, 1, 2, 112, 2, 1280, 0xFD),    # 2DD 640KB
    (737280,  512,  2, 1, 2, 112, 3, 1440, 0xFD),    # 2DD 720KB
    (1474560, 512,  1, 1, 2, 224, 9, 2880, 0xF0),    # 1.44MB
]


class FileEntry:
    """Represents a file or directory in the FAT filesystem."""

    def __init__(self, name, ext, attr, cluster, size, date, time_val, raw_name=None):
        self.raw_name = raw_name or name
        self.name = name.rstrip()
        self.ext = ext.rstrip()
        self.attr = attr
        self.cluster = cluster
        self.size = size
        self.date = date
        self.time = time_val
        self.children = {}

    @property
    def is_directory(self):
        return bool(self.attr & ATTR_DIRECTORY)

    @property
    def is_volume_label(self):
        return bool(self.attr & ATTR_VOLUME_ID)

    @property
    def display_name(self):
        if self.is_directory or not self.ext:
            return self.name
        return f"{self.name}.{self.ext}"

    @property
    def datetime(self):
        try:
            day = self.date & 0x1F
            month = (self.date >> 5) & 0x0F
            year = ((self.date >> 9) & 0x7F) + 1980
            sec = (self.time & 0x1F) * 2
            minute = (self.time >> 5) & 0x3F
            hour = (self.time >> 11) & 0x1F
            if day == 0: day = 1
            if month == 0: month = 1
            return datetime(year, month, day, hour, minute, min(sec, 59))
        except (ValueError, OverflowError):
            return datetime(1980, 1, 1)

    def __repr__(self):
        kind = "DIR" if self.is_directory else "FILE"
        return f"<{kind} {self.display_name} size={self.size} cluster={self.cluster}>"


class FATFilesystem:
    """
    Parses a FAT12 or FAT16 filesystem from a disk image.

    Handles sector size mismatches: the BPB may declare 1024-byte sectors
    while the disk image parser reports a different native sector size.
    All reads go through _read_fs_bytes() which works in absolute byte
    offsets and is independent of the disk's sector granularity.
    """

    def __init__(self, disk_image):
        self.disk = disk_image
        self._image_total_bytes = disk_image.total_sectors * disk_image.sector_size
        self._partition_byte_offset = 0
        self._parse_bpb()
        self._load_fat()
        self._build_root()

    # ── Byte-level read layer ────────────────────────────────────────

    def _read_fs_bytes(self, byte_offset, byte_length):
        if byte_length == 0:
            return b''
        byte_offset += self._partition_byte_offset
        ds = self.disk.sector_size
        first_disk_sector = byte_offset // ds
        last_disk_sector = (byte_offset + byte_length - 1) // ds
        count = last_disk_sector - first_disk_sector + 1
        raw = self.disk.read_sectors(first_disk_sector, count)
        local_start = byte_offset % ds
        return raw[local_start:local_start + byte_length]

    def _read_fs_sectors(self, fs_sector, count):
        byte_offset = fs_sector * self.bytes_per_sector
        byte_length = count * self.bytes_per_sector
        return self._read_fs_bytes(byte_offset, byte_length)

    # ── Byte-level write layer ───────────────────────────────────────

    def _write_fs_bytes(self, byte_offset, data):
        """Write *data* at *byte_offset* in the image, handling sector
        boundaries and partial-sector writes transparently."""
        if not data:
            return
        byte_offset += self._partition_byte_offset
        ds = self.disk.sector_size
        pos = 0
        while pos < len(data):
            current_byte = byte_offset + pos
            sector_num = current_byte // ds
            sector_off = current_byte % ds
            can_write = min(ds - sector_off, len(data) - pos)

            if sector_off == 0 and can_write == ds:
                # Full-sector write — no read needed.
                self.disk.write_sector(sector_num, data[pos:pos + ds])
            else:
                # Partial sector: read-modify-write.
                sector_data = bytearray(self.disk.read_sector(sector_num))
                sector_data[sector_off:sector_off + can_write] = (
                    data[pos:pos + can_write]
                )
                self.disk.write_sector(sector_num, bytes(sector_data))
            pos += can_write

    def _write_fs_sectors(self, fs_sector, data):
        """Write *data* starting at filesystem sector *fs_sector*."""
        self._write_fs_bytes(fs_sector * self.bytes_per_sector, data)

    # ── BPB parsing with validation ──────────────────────────────────

    def _bpb_is_sane(self, bps, spc, reserved, nfats, root_ents, fat_sz, total, **_kw):
        if bps not in (128, 256, 512, 1024, 2048, 4096):
            return False
        if spc == 0 or spc > 128 or (spc & (spc - 1)) != 0:
            return False
        if reserved == 0 or reserved > 64:
            return False
        if nfats == 0 or nfats > 4:
            return False
        if root_ents == 0 or root_ents > 4096:
            return False
        if fat_sz == 0 or fat_sz > 256:
            return False
        if total == 0:
            return False
        root_dir_sects = (root_ents * 32 + bps - 1) // bps
        first_data = reserved + nfats * fat_sz + root_dir_sects
        if first_data >= total:
            return False
        bpb_bytes = total * bps
        if bpb_bytes > self._image_total_bytes * 2:
            return False
        return True

    def _try_partitioned_disk(self):
        """Detect a partitioned disk and locate the FAT partition.

        Delegates partition-table detection to ``partition.py`` and
        then probes each discovered partition for a valid FAT BPB.
        Returns True on success.
        """
        parts = detect_partitions(self.disk)
        for part in parts:
            log.info(f"Probing {part}")
            if self._try_bpb_at(part.byte_offset, part.byte_size):
                return True
        return False

    @staticmethod
    def _read_bpb_fields(boot):
        """Extract raw BPB fields from a boot sector.  Returns a dict."""
        bps = struct.unpack_from('<H', boot, 0x0B)[0]
        total16 = struct.unpack_from('<H', boot, 0x13)[0]
        total = total16
        if total == 0 and len(boot) >= 0x24:
            total = struct.unpack_from('<I', boot, 0x20)[0]
        return {
            'bps':       bps,
            'spc':       boot[0x0D],
            'reserved':  struct.unpack_from('<H', boot, 0x0E)[0],
            'nfats':     boot[0x10],
            'root_ents': struct.unpack_from('<H', boot, 0x11)[0],
            'media':     boot[0x15],
            'fat_sz':    struct.unpack_from('<H', boot, 0x16)[0],
            'total':     total,
        }

    def _apply_bpb(self, f, fallback_total=0):
        """Store validated BPB fields dict *f* into instance attributes."""
        self.bytes_per_sector = f['bps']
        self.sectors_per_cluster = f['spc']
        self.reserved_sectors = f['reserved']
        self.num_fats = f['nfats']
        self.root_entry_count = f['root_ents']
        self.fat_size_16 = f['fat_sz']
        self.media_descriptor = f['media']
        self.total_sectors = f['total'] if f['total'] > 0 else fallback_total

    def _try_bpb_at(self, part_byte_off, part_size_hint=0):
        """Try to read and validate a BPB at *part_byte_off*.

        Sets ``_partition_byte_offset``, ``_image_total_bytes`` and all
        BPB fields on success.  Returns True/False.
        """
        saved_offset = self._partition_byte_offset
        saved_total = self._image_total_bytes

        self._partition_byte_offset = part_byte_off
        part_total_bytes = (
            part_size_hint if part_size_hint
            else saved_total - part_byte_off
        )
        self._image_total_bytes = part_total_bytes

        try:
            pbr = self._read_fs_bytes(0, min(1024, part_total_bytes))
        except Exception:
            self._partition_byte_offset = saved_offset
            self._image_total_bytes = saved_total
            return False

        f = self._read_bpb_fields(pbr)
        if self._bpb_is_sane(**f):
            log.info(f"BPB valid at byte offset 0x{part_byte_off:X}")
            self._apply_bpb(f, part_total_bytes // f['bps'] if f['bps'] else 0)
            return True

        self._partition_byte_offset = saved_offset
        self._image_total_bytes = saved_total
        return False

    def _parse_bpb(self):
        boot = self._read_fs_bytes(0, min(1024, self._image_total_bytes))
        f = self._read_bpb_fields(boot)

        log.info(
            f"BPB raw: bps={f['bps']} spc={f['spc']} reserved={f['reserved']} "
            f"nfats={f['nfats']} root_ents={f['root_ents']} fat_sz={f['fat_sz']} "
            f"total={f['total']} media=0x{f['media']:02X}"
        )
        log.info(
            f"Image: {self._image_total_bytes} bytes, "
            f"disk reports {self.disk.total_sectors} sectors × {self.disk.sector_size}B"
        )

        if self._bpb_is_sane(**f):
            log.info("BPB valid")
            self._apply_bpb(f, self._image_total_bytes // f['bps'] if f['bps'] else 0)
        elif self._try_partitioned_disk():
            # Re-read boot sector from the partition for volume label.
            boot = self._read_fs_bytes(0, min(1024, self._image_total_bytes))
        else:
            log.warning("BPB invalid, trying known PC-98 geometries")
            self._apply_geometry_fallback()

        # Compute layout (all in BPB sector units)
        self.root_dir_sectors = (
            (self.root_entry_count * 32 + self.bytes_per_sector - 1)
            // self.bytes_per_sector
        )
        self.first_fat_sector = self.reserved_sectors
        self.first_root_sector = (
            self.first_fat_sector + self.num_fats * self.fat_size_16
        )
        self.first_data_sector = self.first_root_sector + self.root_dir_sectors

        data_sectors = self.total_sectors - self.first_data_sector
        if data_sectors > 0 and self.sectors_per_cluster > 0:
            self.total_clusters = data_sectors // self.sectors_per_cluster
        else:
            self.total_clusters = 0
        self.fat_type = 12 if self.total_clusters < 4085 else 16

        self.volume_label = ""
        try:
            self.volume_label = boot[0x2B:0x36].decode(
                'shift_jis', errors='replace'
            ).rstrip().rstrip('\x00')
        except Exception:
            pass

        log.info(
            f"Layout: FAT{self.fat_type}, "
            f"root at FS-sector {self.first_root_sector} "
            f"(byte offset 0x{self.first_root_sector * self.bytes_per_sector:X}), "
            f"data at FS-sector {self.first_data_sector} "
            f"(byte offset 0x{self.first_data_sector * self.bytes_per_sector:X}), "
            f"{self.total_clusters} clusters"
        )

        self._validate_fat_header()

    def _apply_geometry_fallback(self):
        for (geo_bytes, geo_bps, geo_spc, geo_res, geo_nfats,
             geo_root, geo_fat, geo_total, geo_media) in PC98_KNOWN_GEOMETRIES:
            if abs(self._image_total_bytes - geo_bytes) < 4096:
                log.info(
                    f"Matched geometry: {geo_bytes} bytes, "
                    f"{geo_bps}B sectors, {geo_total} total"
                )
                self.bytes_per_sector = geo_bps
                self.sectors_per_cluster = geo_spc
                self.reserved_sectors = geo_res
                self.num_fats = geo_nfats
                self.root_entry_count = geo_root
                self.fat_size_16 = geo_fat
                self.media_descriptor = geo_media
                self.total_sectors = geo_total
                return

        log.warning("No geometry match — using default PC-98 2HD layout")
        bps = 1024 if self._image_total_bytes % 1024 == 0 else 512
        self.bytes_per_sector = bps
        self.sectors_per_cluster = 1
        self.reserved_sectors = 1
        self.num_fats = 2
        self.root_entry_count = 192
        self.fat_size_16 = 2
        self.media_descriptor = 0xFE
        self.total_sectors = self._image_total_bytes // bps

    def _validate_fat_header(self):
        try:
            fat_data = self._read_fs_sectors(self.first_fat_sector, 1)
            first_byte = fat_data[0]
            valid_media = {0xF0, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF}
            if first_byte not in valid_media:
                log.warning(
                    f"FAT first byte 0x{first_byte:02X} not a valid media "
                    f"descriptor — may not be FAT"
                )
            else:
                log.info(f"FAT header valid, media byte 0x{first_byte:02X}")
        except Exception as e:
            log.warning(f"Could not read FAT header: {e}")

    # ── FAT table ────────────────────────────────────────────────────

    def _load_fat(self):
        self._fat_data = self._read_fs_sectors(
            self.first_fat_sector, self.fat_size_16
        )

    def get_fat_entry(self, cluster):
        if self.fat_type == 12:
            return self._get_fat12_entry(cluster)
        return self._get_fat16_entry(cluster)

    def _get_fat12_entry(self, cluster):
        offset = cluster + (cluster // 2)
        if offset + 1 >= len(self._fat_data):
            return 0xFFF
        word = struct.unpack_from('<H', self._fat_data, offset)[0]
        return (word >> 4) if (cluster & 1) else (word & 0x0FFF)

    def _get_fat16_entry(self, cluster):
        offset = cluster * 2
        if offset + 1 >= len(self._fat_data):
            return 0xFFFF
        return struct.unpack_from('<H', self._fat_data, offset)[0]

    def get_cluster_chain(self, start_cluster):
        if start_cluster < 2:
            return []
        chain = []
        cluster = start_cluster
        eoc = FAT12_EOC if self.fat_type == 12 else FAT16_EOC
        bad = FAT12_BAD if self.fat_type == 12 else FAT16_BAD
        max_cl = self.total_clusters + 2
        visited = set()
        while cluster >= 2 and cluster < eoc and cluster != bad:
            if cluster in visited or cluster >= max_cl:
                break
            visited.add(cluster)
            chain.append(cluster)
            cluster = self.get_fat_entry(cluster)
        return chain

    def cluster_to_fs_sector(self, cluster):
        return self.first_data_sector + (cluster - 2) * self.sectors_per_cluster

    def read_cluster(self, cluster):
        return self._read_fs_sectors(
            self.cluster_to_fs_sector(cluster), self.sectors_per_cluster
        )

    def read_file(self, entry):
        if entry.size == 0 and not entry.is_directory:
            return b''
        chain = self.get_cluster_chain(entry.cluster)
        data = bytearray()
        for cl in chain:
            data.extend(self.read_cluster(cl))
        return bytes(data[:entry.size]) if not entry.is_directory else bytes(data)

    # ── Directory parsing ────────────────────────────────────────────

    def _parse_dir_entries(self, raw_data):
        entries = []
        offset = 0
        while offset + 32 <= len(raw_data):
            entry_data = raw_data[offset:offset + 32]
            offset += 32

            first_byte = entry_data[0]
            if first_byte == 0x00:
                break
            if first_byte == 0xE5:
                continue

            attr = entry_data[11]
            if (attr & ATTR_LFN) == ATTR_LFN:
                continue
            if first_byte < 0x20 and first_byte != 0x05:
                continue
            if attr & 0xC0:
                continue

            try:
                name = entry_data[0:8].decode('shift_jis', errors='replace')
                ext = entry_data[8:11].decode('shift_jis', errors='replace')
            except Exception:
                name = entry_data[0:8].decode('ascii', errors='replace')
                ext = entry_data[8:11].decode('ascii', errors='replace')

            if entry_data[0] == 0x05:
                name = chr(0xE5) + name[1:]

            time_val = struct.unpack_from('<H', entry_data, 22)[0]
            date_val = struct.unpack_from('<H', entry_data, 24)[0]
            cluster = struct.unpack_from('<H', entry_data, 26)[0]
            size = struct.unpack_from('<I', entry_data, 28)[0]

            entry = FileEntry(name, ext, attr, cluster, size, date_val, time_val,
                              raw_name=entry_data[0:11])

            if entry.is_volume_label:
                if not self.volume_label:
                    self.volume_label = entry.display_name
                continue

            entries.append(entry)

        return entries

    def _build_root(self):
        log.info(
            f"Reading root dir: FS-sector {self.first_root_sector}, "
            f"{self.root_dir_sectors} FS-sectors, "
            f"byte offset 0x{self.first_root_sector * self.bytes_per_sector:X}"
        )
        root_data = self._read_fs_sectors(
            self.first_root_sector, self.root_dir_sectors
        )

        self.root = FileEntry("", "", ATTR_DIRECTORY, 0, 0, 0, 0)
        root_entries = self._parse_dir_entries(root_data)

        log.info(f"Found {len(root_entries)} root directory entries")
        for e in root_entries:
            log.info(f"  {e}")

        for e in root_entries:
            self.root.children[e.display_name.upper()] = e

        self._parse_subdirs(self.root, depth=0)

    def _parse_subdirs(self, dir_entry, depth=0):
        if depth > 20:
            return
        for name, entry in list(dir_entry.children.items()):
            if entry.is_directory and entry.name not in ('.', '..'):
                if entry.cluster >= 2:
                    try:
                        dir_data = self.read_file(entry)
                        sub_entries = self._parse_dir_entries(dir_data)
                        for se in sub_entries:
                            if se.name not in ('.', '..'):
                                entry.children[se.display_name.upper()] = se
                        self._parse_subdirs(entry, depth + 1)
                    except Exception:
                        pass

    # ── Path resolution and walking ──────────────────────────────────

    def resolve_path(self, path):
        path = path.replace('\\', '/')
        parts = [p for p in path.split('/') if p]
        current = self.root
        for part in parts:
            if not current.is_directory:
                return None
            if part.upper() not in current.children:
                return None
            current = current.children[part.upper()]
        return current

    def list_dir(self, path='/'):
        entry = self.resolve_path(path)
        if entry is None or not entry.is_directory:
            return None
        return list(entry.children.values())

    def walk(self, path='/', prefix=''):
        entries = self.list_dir(path)
        if entries is None:
            return
        for e in entries:
            if e.name in ('.', '..'):
                continue
            full = f"{prefix}/{e.display_name}"
            yield full, e
            if e.is_directory:
                yield from self.walk(full, full)

    # =================================================================
    #  FAT WRITE-BACK
    # =================================================================

    @staticmethod
    def _filename_to_83(filename, is_dir=False):
        """Convert a host filename to an (name8, ext3) pair of bytes.

        Names are upper-cased and padded with spaces.  Characters that
        are illegal in 8.3 are replaced with underscores.
        """
        # Characters forbidden in 8.3 names (plus space).
        _INVALID = set(' "+,./;:<=>?[\\]|*')

        def _clean(s):
            out = []
            for ch in s:
                if ord(ch) < 0x20 or ch in _INVALID:
                    out.append('_')
                else:
                    out.append(ch)
            return ''.join(out)

        if is_dir:
            name = _clean(filename.upper())[:8]
            return name.ljust(8).encode('ascii', errors='replace'), \
                   b'   '

        # Split on the *last* dot.
        if '.' in filename:
            base, ext = filename.rsplit('.', 1)
        else:
            base, ext = filename, ''

        base = _clean(base.upper())[:8]
        ext  = _clean(ext.upper())[:3]

        return base.ljust(8).encode('ascii', errors='replace'), \
               ext.ljust(3).encode('ascii', errors='replace')

    @staticmethod
    def _unique_83(name8, ext3, used_names):
        """If *name8*+*ext3* already exists in *used_names*, mangle the
        base with a ``~N`` tail until it's unique.  Returns the
        (possibly mangled) pair and adds it to *used_names*.
        """
        key = name8 + ext3
        if key not in used_names:
            used_names.add(key)
            return name8, ext3

        base = name8.rstrip(b' ')
        for n in range(1, 1000):
            suffix = f"~{n}".encode('ascii')
            max_base = 8 - len(suffix)
            mangled = base[:max_base] + suffix
            mangled = mangled.ljust(8)
            key = mangled + ext3
            if key not in used_names:
                used_names.add(key)
                return mangled, ext3
        raise RuntimeError("Cannot generate unique 8.3 name")

    @staticmethod
    def _make_dir_entry(name8: bytes, ext3: bytes, attr, cluster,
                        size, mtime):
        """Build one 32-byte FAT directory entry."""
        entry = bytearray(32)
        entry[0:8] = name8[:8]
        entry[8:11] = ext3[:3]
        entry[11] = attr

        if isinstance(mtime, datetime):
            fat_time = ((mtime.hour & 0x1F) << 11 |
                        (mtime.minute & 0x3F) << 5 |
                        (mtime.second // 2) & 0x1F)
            fat_date = (((mtime.year - 1980) & 0x7F) << 9 |
                        (mtime.month & 0x0F) << 5 |
                        (mtime.day & 0x1F))
        else:
            fat_time = 0
            fat_date = 0x0021  # 1980-01-01

        struct.pack_into('<H', entry, 22, fat_time)
        struct.pack_into('<H', entry, 24, fat_date)
        struct.pack_into('<H', entry, 26, cluster & 0xFFFF)
        struct.pack_into('<I', entry, 28, size & 0xFFFFFFFF)
        return bytes(entry)

    def _build_fat_bytes(self, fat_table):
        """Serialise the in-memory *fat_table* list to on-disk bytes."""
        fat_bytes_len = self.fat_size_16 * self.bytes_per_sector
        buf = bytearray(fat_bytes_len)

        if self.fat_type == 12:
            for i, val in enumerate(fat_table):
                offset = i + (i // 2)
                if offset + 1 >= len(buf):
                    break
                word = struct.unpack_from('<H', buf, offset)[0]
                if i & 1:
                    word = (word & 0x000F) | ((val & 0x0FFF) << 4)
                else:
                    word = (word & 0xF000) | (val & 0x0FFF)
                struct.pack_into('<H', buf, offset, word)
        else:
            for i, val in enumerate(fat_table):
                offset = i * 2
                if offset + 1 >= len(buf):
                    break
                struct.pack_into('<H', buf, offset, val & 0xFFFF)

        return bytes(buf)

    # ── Cluster allocator helpers (used only during write-back) ──────

    @staticmethod
    def _alloc_cluster(fat, next_free):
        """Allocate one free cluster, advancing *next_free* (a 1-element
        list used as a mutable counter).  Raises RuntimeError if full."""
        while next_free[0] < len(fat):
            if fat[next_free[0]] == 0:
                c = next_free[0]
                next_free[0] += 1
                return c
            next_free[0] += 1
        raise RuntimeError("Disk full — not enough free clusters")

    def _alloc_chain(self, fat, next_free, num_clusters):
        """Allocate a chain of *num_clusters* and link them in *fat*.
        Returns the list of cluster numbers."""
        eoc = 0xFFF if self.fat_type == 12 else 0xFFFF
        chain = []
        for _ in range(num_clusters):
            chain.append(self._alloc_cluster(fat, next_free))
        for i in range(len(chain) - 1):
            fat[chain[i]] = chain[i + 1]
        if chain:
            fat[chain[-1]] = eoc
        return chain

    def _write_to_clusters(self, chain, data, cluster_size):
        """Write *data* into the given cluster *chain*."""
        for i, cluster in enumerate(chain):
            offset = i * cluster_size
            chunk = data[offset:offset + cluster_size]
            if len(chunk) < cluster_size:
                chunk = chunk + b'\x00' * (cluster_size - len(chunk))
            fs_sector = self.cluster_to_fs_sector(cluster)
            self._write_fs_bytes(fs_sector * self.bytes_per_sector, chunk)

    # ── The main write-back entry point ──────────────────────────────

    def write_back_from_directory(self, dir_path, save_path=None):
        """Rebuild the FAT filesystem from the real directory at
        *dir_path*, then save the image.

        * The boot sector / BPB is left untouched.
        * The FAT tables, root directory, and data area are rebuilt from
          scratch based on the files currently present under *dir_path*.
        * If *save_path* is given the image is saved there; otherwise
          the original file is overwritten.

        Returns a ``(files_written, dirs_written)`` tuple.
        """
        cluster_size = self.sectors_per_cluster * self.bytes_per_sector
        max_cluster = self.total_clusters + 2

        # 1. Initialise in-memory FAT.
        fat = [0] * max_cluster
        if self.fat_type == 12:
            fat[0] = 0xF00 | (self.media_descriptor & 0xFF)
            fat[1] = 0xFFF
        else:
            fat[0] = 0xFF00 | (self.media_descriptor & 0xFF)
            fat[1] = 0xFFFF
        next_free = [2]

        # 2. Zero out root directory area.
        root_byte_off = self.first_root_sector * self.bytes_per_sector
        root_byte_len = self.root_dir_sectors * self.bytes_per_sector
        self._write_fs_bytes(root_byte_off, b'\x00' * root_byte_len)

        # 3. Zero out data area.
        data_byte_off = self.first_data_sector * self.bytes_per_sector
        data_byte_len = (
            (self.total_sectors - self.first_data_sector) *
            self.bytes_per_sector
        )
        if data_byte_len > 0:
            # Write in 64 KB chunks to keep memory reasonable.
            CHUNK = 65536
            written = 0
            while written < data_byte_len:
                n = min(CHUNK, data_byte_len - written)
                self._write_fs_bytes(data_byte_off + written, b'\x00' * n)
                written += n

        # 4. Walk host directory and write contents.
        counters = {'files': 0, 'dirs': 0}

        def _process_dir(real_path, parent_cluster, is_root):
            """Process one real directory level.  Returns a list of
            32-byte entry blocks ready to write into a directory area."""
            entries = []
            used_names = set()

            try:
                items = sorted(os.listdir(real_path))
            except OSError as exc:
                log.warning(f"Cannot list {real_path}: {exc}")
                return entries

            for item_name in items:
                item_path = os.path.join(real_path, item_name)

                try:
                    mtime = datetime.fromtimestamp(
                        os.path.getmtime(item_path))
                except OSError:
                    mtime = datetime(1980, 1, 1)

                if os.path.isdir(item_path):
                    # ── Subdirectory ─────────────────────────────────
                    name8, ext3 = self._filename_to_83(item_name,
                                                       is_dir=True)
                    name8, ext3 = self._unique_83(name8, ext3,
                                                   used_names)

                    # Pre-allocate one cluster (expand later if needed).
                    eoc = 0xFFF if self.fat_type == 12 else 0xFFFF
                    dir_cluster = self._alloc_cluster(fat, next_free)
                    fat[dir_cluster] = eoc

                    # Recurse to get child entries.
                    child_entries = _process_dir(
                        item_path, dir_cluster, False)

                    # Build the on-disk directory data: . , .. , children.
                    dot = self._make_dir_entry(
                        b'.       ', b'   ', ATTR_DIRECTORY,
                        dir_cluster, 0, mtime)
                    dotdot = self._make_dir_entry(
                        b'..      ', b'   ', ATTR_DIRECTORY,
                        parent_cluster if not is_root else 0, 0, mtime)

                    dir_data = bytearray(dot + dotdot)
                    for ce in child_entries:
                        dir_data.extend(ce)

                    # Pad to cluster boundary.
                    remainder = len(dir_data) % cluster_size
                    if remainder:
                        dir_data.extend(
                            b'\x00' * (cluster_size - remainder))

                    # Allocate more clusters if the directory grew.
                    num_cl = len(dir_data) // cluster_size
                    chain = [dir_cluster]
                    for _ in range(num_cl - 1):
                        c = self._alloc_cluster(fat, next_free)
                        fat[chain[-1]] = c
                        fat[c] = eoc
                        chain.append(c)

                    self._write_to_clusters(chain, bytes(dir_data),
                                            cluster_size)

                    entry = self._make_dir_entry(
                        name8, ext3, ATTR_DIRECTORY,
                        dir_cluster, 0, mtime)
                    entries.append(entry)
                    counters['dirs'] += 1

                elif os.path.isfile(item_path):
                    # ── Regular file ─────────────────────────────────
                    name8, ext3 = self._filename_to_83(item_name)
                    name8, ext3 = self._unique_83(name8, ext3,
                                                   used_names)
                    try:
                        with open(item_path, 'rb') as fh:
                            file_data = fh.read()
                    except OSError as exc:
                        log.warning(
                            f"Cannot read {item_path}: {exc}")
                        continue

                    file_size = len(file_data)
                    if file_size > 0:
                        num_cl = (
                            (file_size + cluster_size - 1) //
                            cluster_size
                        )
                        chain = self._alloc_chain(
                            fat, next_free, num_cl)
                        self._write_to_clusters(
                            chain, file_data, cluster_size)
                        first_cluster = chain[0]
                    else:
                        first_cluster = 0

                    entry = self._make_dir_entry(
                        name8, ext3, ATTR_ARCHIVE,
                        first_cluster, file_size, mtime)
                    entries.append(entry)
                    counters['files'] += 1

            return entries

        root_entries = _process_dir(dir_path, 0, True)

        # 5. Write volume label back into the root directory if we had one.
        vol_entries = []
        vol_clean = (self.volume_label or '').strip().strip('\x00')
        if vol_clean:
            vol_name = vol_clean.upper()[:11].ljust(11)
            vol_entry = self._make_dir_entry(
                vol_name[:8].encode('ascii', errors='replace'),
                vol_name[8:11].encode('ascii', errors='replace'),
                ATTR_VOLUME_ID, 0, 0, datetime.now())
            vol_entries.append(vol_entry)

        # Combine and check root-directory capacity.
        all_root = vol_entries + root_entries
        max_root_bytes = self.root_entry_count * 32
        root_data = b''.join(all_root)
        if len(root_data) > max_root_bytes:
            raise RuntimeError(
                f"Root directory overflow: {len(all_root)} entries, "
                f"max {self.root_entry_count}"
            )
        self._write_fs_bytes(root_byte_off, root_data)

        # 6. Write FAT tables (all copies).
        fat_data = self._build_fat_bytes(fat)
        for i in range(self.num_fats):
            fat_sector = self.first_fat_sector + i * self.fat_size_16
            self._write_fs_bytes(
                fat_sector * self.bytes_per_sector, fat_data)

        # 7. Save image file.
        self.disk.save(save_path)

        # 8. Reload in-memory FAT and directory tree so subsequent
        #    reads reflect the new state.
        self._load_fat()
        self._build_root()

        log.info(
            f"Write-back complete: {counters['files']} files, "
            f"{counters['dirs']} directories"
        )
        return counters['files'], counters['dirs']