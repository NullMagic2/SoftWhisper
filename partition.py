"""
Partition table detection for PC-98 disk images.

Supports:
  - IBM-PC MBR (Master Boot Record)
  - PC-98 IPL (Initial Program Loader) partition table

Each detector reads the raw disk sectors and returns a list of
``PartitionEntry`` objects describing the partitions found.  The
filesystem layer (``fat_fs.py``) then probes each partition for a
valid BPB.

Adding a new scheme
-------------------
1. Write a ``detect_<scheme>(disk_image)`` function that returns a
   list of ``PartitionEntry``.
2. Register it in ``DETECTORS`` at the bottom of this file.
3. ``detect_partitions()`` will call it automatically.
"""

import struct
import logging

log = logging.getLogger("pc98mount.partition")


# ── Partition entry ──────────────────────────────────────────────

class PartitionEntry:
    """One partition found on a disk image."""

    __slots__ = (
        'index', 'scheme', 'type_id', 'name',
        'byte_offset', 'byte_size',
    )

    def __init__(self, *, index, scheme, type_id, name,
                 byte_offset, byte_size):
        self.index = index
        self.scheme = scheme        # "MBR" or "PC-98"
        self.type_id = type_id      # e.g. 0x06 for FAT16 (MBR)
        self.name = name            # human-readable label
        self.byte_offset = byte_offset
        self.byte_size = byte_size  # 0 = extends to end of image

    def __repr__(self):
        size_str = (f"{self.byte_size:,}" if self.byte_size
                    else "to end")
        return (
            f"<Partition {self.index} [{self.scheme}] "
            f"type=0x{self.type_id:02X} \"{self.name}\" "
            f"offset=0x{self.byte_offset:X} size={size_str}>"
        )


# ── IBM-PC MBR ──────────────────────────────────────────────────

def detect_mbr(disk_image):
    """Detect standard IBM-PC MBR partitions.

    The MBR signature 0x55AA sits at byte offset 0x1FE of sector 0.
    Four 16-byte partition entries live at 0x1BE–0x1FD.
    """
    if disk_image.total_sectors < 1:
        return []
    boot = disk_image.read_sector(0)
    if len(boot) < 512:
        return []
    if boot[0x1FE] != 0x55 or boot[0x1FF] != 0xAA:
        return []

    ds = disk_image.sector_size
    image_bytes = disk_image.total_sectors * ds
    partitions = []

    for i in range(4):
        off = 0x1BE + i * 16
        part_type = boot[off + 4]
        if part_type == 0:
            continue
        lba_start = struct.unpack_from('<I', boot, off + 8)[0]
        lba_size = struct.unpack_from('<I', boot, off + 12)[0]
        if lba_start == 0:
            continue

        byte_offset = lba_start * ds
        if byte_offset >= image_bytes:
            continue

        byte_size = lba_size * ds if lba_size else 0

        log.info(
            f"MBR partition {i}: type=0x{part_type:02X} "
            f"LBA_start={lba_start} LBA_size={lba_size}"
        )

        partitions.append(PartitionEntry(
            index=i,
            scheme="MBR",
            type_id=part_type,
            name=_mbr_type_name(part_type),
            byte_offset=byte_offset,
            byte_size=byte_size,
        ))

    return partitions


_MBR_TYPE_NAMES = {
    0x01: "FAT12",
    0x04: "FAT16 <32M",
    0x06: "FAT16",
    0x0B: "FAT32 CHS",
    0x0C: "FAT32 LBA",
    0x0E: "FAT16 LBA",
    0x0F: "Extended LBA",
}


def _mbr_type_name(type_id):
    return _MBR_TYPE_NAMES.get(type_id, f"type 0x{type_id:02X}")


# ── PC-98 IPL partition table ───────────────────────────────────

def detect_pc98(disk_image):
    """Detect PC-98 IPL partition table.

    PC-98 hard disks reserve cylinder 0 for the IPL boot code and a
    proprietary partition table at sector 1.  Each entry is 32 bytes
    (up to 16 partitions).  The actual FAT partition typically begins
    at cylinder 1.

    Detection heuristics:
      - Sector 0 contains "IPL1" at offset 3, or
      - Sector 0 has the PC-98 0x55AA signature at offset 0xFE
        (as opposed to the IBM 0x1FE location).
    """
    if disk_image.total_sectors < 2:
        return []
    boot = disk_image.read_sector(0)
    if len(boot) < 256:
        return []

    is_pc98 = (
        boot[3:7] == b'IPL1'
        or (len(boot) >= 0x100
            and boot[0xFE] == 0x55 and boot[0xFF] == 0xAA)
    )
    if not is_pc98:
        return []

    log.info("PC-98 IPL detected")

    ds = disk_image.sector_size
    image_bytes = disk_image.total_sectors * ds

    # Read sector 1 which holds the partition table.
    sec1 = disk_image.read_sector(1)
    if len(sec1) < 512:
        return []

    # Determine disk geometry for CHS → LBA conversion.
    spt = getattr(disk_image, '_spt', 0) or 17
    heads = getattr(disk_image, '_heads', 0) or 8

    partitions = []
    for i in range(16):
        off = i * 32
        entry = sec1[off:off + 32]
        if all(b == 0 for b in entry):
            continue

        boot_flag = entry[0]
        sys_id = entry[1]

        # PC-98 CHS layout: head, sector, cylinder (LE16).
        start_head = entry[8]
        start_sec = entry[9]
        start_cyl = struct.unpack_from('<H', entry, 10)[0]

        end_head = entry[12]
        end_sec = entry[13]
        end_cyl = struct.unpack_from('<H', entry, 14)[0]

        # 16-byte partition name in Shift-JIS.
        name_bytes = entry[16:32]
        try:
            name = name_bytes.decode(
                'shift_jis', errors='replace'
            ).rstrip('\x00').rstrip()
        except Exception:
            name = f"partition {i}"

        lba_start = start_cyl * heads * spt + start_head * spt + start_sec
        lba_end = end_cyl * heads * spt + end_head * spt + end_sec

        byte_offset = lba_start * ds
        if byte_offset >= image_bytes:
            continue
        byte_size = (lba_end - lba_start + 1) * ds if lba_end > lba_start else 0

        log.info(
            f"PC-98 partition {i}: sys=0x{sys_id:02X} "
            f"\"{name}\" CHS {start_cyl}/{start_head}/{start_sec} "
            f"– {end_cyl}/{end_head}/{end_sec} "
            f"(LBA {lba_start}–{lba_end})"
        )

        partitions.append(PartitionEntry(
            index=i,
            scheme="PC-98",
            type_id=sys_id,
            name=name,
            byte_offset=byte_offset,
            byte_size=byte_size,
        ))

    # If the partition table was found but yielded no entries (e.g.
    # unpartitioned or the entries are too unusual to parse), fall
    # back to probing common cylinder-1 offsets.
    if not partitions:
        log.info("PC-98 IPL present but no parseable partition entries; "
                 "trying cylinder 1 heuristic")
        partitions = _pc98_cylinder1_fallback(disk_image, spt, heads)

    return partitions


def _pc98_cylinder1_fallback(disk_image, spt, heads):
    """Return synthetic partition entries for cylinder-1 offsets.

    When the PC-98 partition table is present but can't be parsed
    reliably, we generate candidates at the most common cylinder-1
    boundaries and let the filesystem layer probe each one.
    """
    ds = disk_image.sector_size
    image_bytes = disk_image.total_sectors * ds

    seen = set()
    entries = []
    for s, h in ((spt, heads), (17, 8), (17, 4), (25, 8), (32, 8)):
        lba = s * h
        if lba in seen:
            continue
        seen.add(lba)
        byte_off = lba * ds
        if byte_off >= image_bytes:
            continue
        entries.append(PartitionEntry(
            index=len(entries),
            scheme="PC-98",
            type_id=0,
            name=f"cylinder 1 ({s}×{h})",
            byte_offset=byte_off,
            byte_size=0,
        ))
    return entries


# ── Public API ──────────────────────────────────────────────────

# Detectors are tried in order; the first one that returns a
# non-empty list wins.
DETECTORS = [
    ("MBR",   detect_mbr),
    ("PC-98", detect_pc98),
]


def detect_partitions(disk_image):
    """Auto-detect the partition scheme and return all partitions.

    Returns a list of ``PartitionEntry`` objects.  If no partition
    table is recognised the list is empty (the image is likely an
    unpartitioned floppy or a raw dump).
    """
    for name, detector in DETECTORS:
        try:
            parts = detector(disk_image)
            if parts:
                log.info(
                    f"Partition scheme: {name} "
                    f"({len(parts)} partition(s))"
                )
                return parts
        except Exception as exc:
            log.warning(f"{name} detection failed: {exc}")
    return []