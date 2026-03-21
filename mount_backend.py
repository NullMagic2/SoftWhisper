"""
Mount backends for PC-98 disk images.

Supports Windows and Linux (including WSL).

Windows strategies (tried in order):
  1. VHD (preferred) — creates a virtual hard disk of the correct size,
     formats it, copies files, and mounts as a drive letter.
     Shows correct disk size in Explorer. Requires admin privileges.
  2. subst (fallback) — extracts to a temp directory and uses `subst`
     to assign a drive letter. No admin needed, but Explorer shows the
     host drive's size instead of the image's.

Linux / WSL strategy:
  3. Directory mount — extracts to a directory under a configurable base
     path (default: ~/.local/share/pc98mount/mounts).

All strategies are read-only from the image's perspective.
The ``update`` method on each mount class (and on MountManager) writes
changes made through the file manager back into the disk image.
"""

import atexit
import glob
import os
import sys
import shutil
import subprocess
import tempfile
import time
import logging
from pathlib import Path

log = logging.getLogger("pc98mount.mount")


# =============================================================================
# Platform detection
# =============================================================================

def is_windows():
    return sys.platform == 'win32'


def is_wsl():
    """Detect Windows Subsystem for Linux."""
    if is_windows():
        return False
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except (OSError, IOError):
        return False


_IS_WSL = None


def _cached_is_wsl():
    global _IS_WSL
    if _IS_WSL is None:
        _IS_WSL = is_wsl()
    return _IS_WSL


# =============================================================================
# Helpers
# =============================================================================

def _sanitize_filename(name):
    """Remove or replace characters invalid in filenames."""
    invalid = '<>:"/\\|?*'
    result = ''.join(c if c not in invalid else '_' for c in name)
    result = result.rstrip('. ')
    return result if result else '_unnamed_'


def _extract_fat_to_dir(fat_fs, dir_entry, dest_path, counters):
    """Recursively extract a FAT directory tree to a real directory."""
    for name, entry in dir_entry.children.items():
        if entry.name in ('.', '..'):
            continue

        safe_name = _sanitize_filename(entry.display_name)
        target = os.path.join(dest_path, safe_name)

        if entry.is_directory:
            log.info(f"  DIR:  {safe_name}/")
            os.makedirs(target, exist_ok=True)
            _extract_fat_to_dir(fat_fs, entry, target, counters)
        else:
            try:
                data = fat_fs.read_file(entry)
                with open(target, 'wb') as f:
                    f.write(data)
                counters['files'] += 1
                log.info(f"  FILE: {safe_name} ({len(data)} bytes)")
            except Exception as e:
                counters['errors'] += 1
                log.error(f"  FAIL: {safe_name}: {e}")


def _write_flat_to_dir(disk_image, dest_path, filename="DISK.IMG"):
    """Write the entire raw image as a single file."""
    out_path = os.path.join(dest_path, filename)
    data = disk_image.read_sectors(0, disk_image.total_sectors)
    with open(out_path, 'wb') as f:
        f.write(data)
    log.info(f"Wrote {len(data):,} bytes to {out_path}")


def _write_sectors_to_dir(disk_image, dest_path):
    """Write each sector as an individual file."""
    width = len(str(disk_image.total_sectors - 1))
    for i in range(disk_image.total_sectors):
        filename = f"SECTOR_{i:0{width}d}.BIN"
        data = disk_image.read_sector(i)
        with open(os.path.join(dest_path, filename), 'wb') as f:
            f.write(data)
    log.info(f"Wrote {disk_image.total_sectors} sector files to {dest_path}")


def _is_admin():
    """Check if we're running with admin/root privileges."""
    if is_windows():
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False
    else:
        return os.geteuid() == 0


def _run_diskpart(script_text):
    script_path = os.path.join(tempfile.gettempdir(), 'pc98_diskpart.txt')
    with open(script_path, 'w') as f:
        f.write(script_text)
    try:
        result = subprocess.run(
            ['diskpart', '/s', script_path],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        log.info(f"diskpart output:\n{output}")
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "diskpart timed out"
    except FileNotFoundError:
        return False, "diskpart not found"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def _run_elevated(command, args):
    if not is_windows():
        return False, "Not on Windows"
    import ctypes
    batch_path = os.path.join(tempfile.gettempdir(), 'pc98_elevated.bat')
    output_path = os.path.join(tempfile.gettempdir(), 'pc98_elevated_out.txt')
    done_path = os.path.join(tempfile.gettempdir(), 'pc98_elevated_done.txt')

    for p in (output_path, done_path):
        try:
            os.unlink(p)
        except OSError:
            pass

    with open(batch_path, 'w') as f:
        f.write(f'@echo off\n')
        f.write(f'{command} {args} > "{output_path}" 2>&1\n')
        f.write(f'echo DONE > "{done_path}"\n')

    try:
        ret = ctypes.windll.shell32.ShellExecuteW(
            None, "runas", "cmd.exe",
            f'/c "{batch_path}"',
            None, 0  # SW_HIDE
        )
        if ret <= 32:
            return False, f"ShellExecute failed (code {ret})"
        for _ in range(60):
            time.sleep(0.5)
            if os.path.exists(done_path):
                break
        else:
            return False, "Elevated process timed out"
        output = ""
        if os.path.exists(output_path):
            with open(output_path, 'r', errors='replace') as f:
                output = f.read()
        return True, output
    finally:
        for p in (batch_path, output_path, done_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def open_in_file_manager(path):
    """Open a path in the platform's file manager."""
    if is_windows():
        os.startfile(path)
    elif _cached_is_wsl():
        try:
            result = subprocess.run(
                ['wslpath', '-w', path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                win_path = result.stdout.strip()
                subprocess.Popen(['explorer.exe', win_path])
            else:
                subprocess.Popen(['explorer.exe', path])
        except FileNotFoundError:
            subprocess.Popen(['xdg-open', path])
    else:
        try:
            subprocess.Popen(['xdg-open', path])
        except FileNotFoundError:
            try:
                subprocess.Popen(['open', path])   # macOS
            except FileNotFoundError:
                log.error(
                    "No file manager command found "
                    "(tried xdg-open, open)")


# =============================================================================
# Write-back helpers (flat / sectors modes)
# =============================================================================

def _update_flat_from_dir(disk_image, mount_path, save_path=None):
    """Read DISK.IMG back from the mount point and overwrite the image."""
    img_path = os.path.join(mount_path, "DISK.IMG")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(
            "DISK.IMG not found in the mount point — "
            "cannot write back in flat mode."
        )
    with open(img_path, 'rb') as fh:
        data = fh.read()

    ss = disk_image.sector_size
    sectors_in_file = len(data) // ss
    written = 0
    for lba in range(min(sectors_in_file, disk_image.total_sectors)):
        disk_image.write_sector(lba, data[lba * ss:(lba + 1) * ss])
        written += 1

    disk_image.save(save_path)
    log.info(f"Flat write-back: {written} sectors written")
    return written


def _update_sectors_from_dir(disk_image, mount_path, save_path=None):
    """Read individual SECTOR_NNNN.BIN files back from the mount point."""
    width = len(str(disk_image.total_sectors - 1))
    written = 0
    for lba in range(disk_image.total_sectors):
        filename = f"SECTOR_{lba:0{width}d}.BIN"
        fpath = os.path.join(mount_path, filename)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as fh:
                data = fh.read()
            disk_image.write_sector(lba, data)
            written += 1

    disk_image.save(save_path)
    log.info(f"Sector write-back: {written} sector files written")
    return written


# =============================================================================
# Strategy 1: VHD Mount (Windows only, correct disk size, needs admin)
# =============================================================================

class VHDMount:
    """
    Creates a VHD (Virtual Hard Disk), formats it to the correct size,
    copies files, and assigns a drive letter. Explorer shows the real size.
    Windows only.
    """

    def __init__(self, drive_letter):
        self.drive_letter = drive_letter.upper().rstrip(':')
        self._vhd_path = None
        self._temp_dir = None
        self._mounted = False
        self._extract_count = 0
        self._extract_errors = 0

    @property
    def mount_point(self):
        return f"{self.drive_letter}:"

    @property
    def is_mounted(self):
        return self._mounted

    def mount_fat(self, fat_fs, image_size_bytes):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98vhd_")
        staging = os.path.join(self._temp_dir, "staging")
        os.makedirs(staging)
        counters = {'files': 0, 'errors': 0}
        _extract_fat_to_dir(fat_fs, fat_fs.root, staging, counters)
        self._extract_count = counters['files']
        self._extract_errors = counters['errors']
        log.info(
            f"Extracted {counters['files']} files, "
            f"{counters['errors']} errors"
        )
        self._create_and_mount_vhd(image_size_bytes, staging)

    def mount_flat(self, disk_image):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98vhd_")
        staging = os.path.join(self._temp_dir, "staging")
        os.makedirs(staging)
        _write_flat_to_dir(disk_image, staging)
        size = disk_image.total_sectors * disk_image.sector_size
        self._create_and_mount_vhd(size, staging)

    def mount_sectors(self, disk_image):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98vhd_")
        staging = os.path.join(self._temp_dir, "staging")
        os.makedirs(staging)
        _write_sectors_to_dir(disk_image, staging)
        size = disk_image.total_sectors * disk_image.sector_size
        self._create_and_mount_vhd(size, staging)

    def _create_and_mount_vhd(self, image_size_bytes, staging_dir):
        vhd_size_mb = max(3, (image_size_bytes // (1024 * 1024)) + 2)
        self._vhd_path = os.path.join(self._temp_dir, "pc98disk.vhd")
        log.info(f"Creating {vhd_size_mb}MB VHD at {self._vhd_path}")

        create_script = (
            f'create vdisk file="{self._vhd_path}" '
            f'maximum={vhd_size_mb} type=fixed\n'
            f'select vdisk file="{self._vhd_path}"\n'
            f'attach vdisk\n'
            f'create partition primary\n'
            f'format fs=fat32 quick label="PC98"\n'
            f'assign letter={self.drive_letter}\n'
        )

        if _is_admin():
            ok, output = _run_diskpart(create_script)
        else:
            script_path = os.path.join(self._temp_dir, 'create.txt')
            with open(script_path, 'w') as f:
                f.write(create_script)
            ok, output = _run_elevated(
                'diskpart', f'/s "{script_path}"')

        if not ok:
            raise RuntimeError(f"Failed to create VHD:\n{output}")

        drive_path = f"{self.drive_letter}:\\"
        for _ in range(20):
            if os.path.exists(drive_path):
                break
            time.sleep(0.3)
        else:
            raise RuntimeError(
                f"Drive {self.drive_letter}: did not appear "
                f"after VHD mount"
            )

        log.info(f"Copying files to {drive_path}")
        self._copy_tree(staging_dir, drive_path)

        try:
            for root, dirs, files in os.walk(drive_path):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    try:
                        os.chmod(fpath, 0o444)
                    except OSError:
                        pass
        except Exception:
            pass

        self._mounted = True
        log.info(f"VHD mounted at {self.drive_letter}:")

    def _copy_tree(self, src, dst):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                os.makedirs(d, exist_ok=True)
                self._copy_tree(s, d)
            else:
                shutil.copy2(s, d)

    def unmount(self):
        if not self._mounted and not self._vhd_path:
            return
        if self._vhd_path and os.path.exists(self._vhd_path):
            detach_script = (
                f'select vdisk file="{self._vhd_path}"\n'
                f'detach vdisk\n'
            )
            if _is_admin():
                _run_diskpart(detach_script)
            else:
                script_path = os.path.join(
                    tempfile.gettempdir(), 'pc98_detach.txt')
                with open(script_path, 'w') as f:
                    f.write(detach_script)
                _run_elevated('diskpart', f'/s "{script_path}"')
                time.sleep(1)

        self._mounted = False
        if self._temp_dir and os.path.exists(self._temp_dir):
            time.sleep(0.5)
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                log.info(f"Cleaned up {self._temp_dir}")
            except Exception as e:
                log.warning(f"Cleanup failed: {e}")
        self._temp_dir = None
        self._vhd_path = None


# =============================================================================
# Strategy 2: Extract + subst (Windows, zero dependencies, no admin)
# =============================================================================

class SubstMount:
    """
    Extracts image contents to a temp directory and uses
    Windows `subst` to map a drive letter to that directory.
    """

    def __init__(self, drive_letter):
        self.drive_letter = drive_letter.upper().rstrip(':')
        self._temp_dir = None
        self._mounted = False
        self._extract_count = 0
        self._extract_errors = 0

    @property
    def mount_point(self):
        return f"{self.drive_letter}:"

    @property
    def is_mounted(self):
        return self._mounted

    def mount_fat(self, fat_fs):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98_")
        counters = {'files': 0, 'errors': 0}
        _extract_fat_to_dir(fat_fs, fat_fs.root, self._temp_dir, counters)
        self._extract_count = counters['files']
        self._extract_errors = counters['errors']
        log.info(
            f"Extracted {counters['files']} files, "
            f"{counters['errors']} errors to {self._temp_dir}"
        )
        if counters['files'] == 0:
            log.warning("No files extracted!")
        self._subst_mount()

    def mount_flat(self, disk_image):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98raw_")
        _write_flat_to_dir(disk_image, self._temp_dir)
        self._subst_mount()

    def mount_sectors(self, disk_image):
        self._temp_dir = tempfile.mkdtemp(prefix="pc98sec_")
        _write_sectors_to_dir(disk_image, self._temp_dir)
        self._subst_mount()

    def _subst_mount(self):
        try:
            subprocess.run(
                ['subst', f'{self.drive_letter}:', '/D'],
                capture_output=True, check=False
            )
            result = subprocess.run(
                ['subst', f'{self.drive_letter}:', self._temp_dir],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"subst failed: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            self._mounted = True
            log.info(
                f"Mounted {self._temp_dir} at "
                f"{self.drive_letter}: via subst"
            )
        except FileNotFoundError:
            raise RuntimeError("'subst' command not found")

    def unmount(self):
        if not self._mounted:
            return
        try:
            subprocess.run(
                ['subst', f'{self.drive_letter}:', '/D'],
                capture_output=True, text=True
            )
        except Exception as e:
            log.warning(f"subst /D failed: {e}")

        self._mounted = False
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                log.info(f"Cleaned up {self._temp_dir}")
            except Exception as e:
                log.warning(f"Cleanup failed: {e}")
        self._temp_dir = None

    @property
    def content_dir(self):
        """The real directory behind the subst alias."""
        return self._temp_dir


# =============================================================================
# Strategy 3: Directory mount (Linux / WSL / any platform)
# =============================================================================

class DirectoryMount:
    """
    Extracts image contents into a directory and exposes that path as the
    mount point.  Works on any platform.
    """

    DEFAULT_BASE = os.path.expanduser("~/.local/share/pc98mount/mounts")

    def __init__(self, mount_path):
        self._mount_path = os.path.abspath(mount_path)
        self._mounted = False
        self._extract_count = 0
        self._extract_errors = 0

    @property
    def mount_point(self):
        return self._mount_path

    @property
    def is_mounted(self):
        return self._mounted

    def mount_fat(self, fat_fs):
        self._ensure_dir()
        counters = {'files': 0, 'errors': 0}
        _extract_fat_to_dir(fat_fs, fat_fs.root, self._mount_path, counters)
        self._extract_count = counters['files']
        self._extract_errors = counters['errors']
        log.info(
            f"Extracted {counters['files']} files, "
            f"{counters['errors']} errors to {self._mount_path}"
        )
        if counters['files'] == 0:
            log.warning("No files extracted!")
        self._mounted = True

    def mount_flat(self, disk_image):
        self._ensure_dir()
        _write_flat_to_dir(disk_image, self._mount_path)
        self._mounted = True

    def mount_sectors(self, disk_image):
        self._ensure_dir()
        _write_sectors_to_dir(disk_image, self._mount_path)
        self._mounted = True

    def unmount(self):
        if not self._mounted:
            return
        self._mounted = False
        if os.path.exists(self._mount_path):
            try:
                shutil.rmtree(self._mount_path, ignore_errors=True)
                log.info(f"Cleaned up {self._mount_path}")
            except Exception as e:
                log.warning(f"Cleanup failed: {e}")

    def _ensure_dir(self):
        if os.path.exists(self._mount_path):
            shutil.rmtree(self._mount_path, ignore_errors=True)
        os.makedirs(self._mount_path, exist_ok=True)

    @classmethod
    def default_base(cls):
        return cls.DEFAULT_BASE


# =============================================================================
# Stale mount cleanup (runs at startup and atexit)
# =============================================================================

def _find_stale_vhd_dirs():
    """Return a list of pc98vhd_* temp directories that still exist."""
    pattern = os.path.join(tempfile.gettempdir(), 'pc98vhd_*')
    return [p for p in glob.glob(pattern) if os.path.isdir(p)]


def _find_stale_subst_dirs():
    """Return a list of pc98_*, pc98raw_*, pc98sec_* temp directories."""
    tmp = tempfile.gettempdir()
    dirs = []
    for prefix in ('pc98_*', 'pc98raw_*', 'pc98sec_*'):
        dirs.extend(p for p in glob.glob(os.path.join(tmp, prefix))
                    if os.path.isdir(p))
    return dirs


def _detach_vhd_file(vhd_path):
    """Try to detach a VHD via diskpart.  Returns True on success."""
    detach_script = (
        f'select vdisk file="{vhd_path}"\n'
        f'detach vdisk\n'
    )
    try:
        if _is_admin():
            ok, output = _run_diskpart(detach_script)
        else:
            ok, output = _run_elevated('diskpart',
                                       f'/s "{_write_temp_script(detach_script)}"')
        return ok
    except Exception as exc:
        log.debug(f"Detach failed for {vhd_path}: {exc}")
        return False


def _write_temp_script(text):
    """Write *text* to a temp file and return its path."""
    path = os.path.join(tempfile.gettempdir(), 'pc98_cleanup.txt')
    with open(path, 'w') as f:
        f.write(text)
    return path


def _is_vhd_attached(vhd_path):
    """Check if a VHD file is locked (i.e. still attached)."""
    try:
        with open(vhd_path, 'ab') as _:
            pass
        return False
    except (PermissionError, OSError):
        return True


def _remove_subst_for_dir(dir_path):
    """If any drive letter is subst'd to *dir_path*, remove it."""
    try:
        result = subprocess.run(
            ['subst'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return
        # Output lines look like:  P:\: => C:\Users\x\AppData\Local\Temp\pc98_abc
        for line in result.stdout.splitlines():
            if '=>' not in line:
                continue
            drive_part, _, target = line.partition('=>')
            target = target.strip().rstrip('\\')
            dir_norm = os.path.normcase(os.path.normpath(dir_path))
            target_norm = os.path.normcase(os.path.normpath(target))
            if target_norm == dir_norm:
                letter = drive_part.strip().rstrip(':\\')
                log.info(f"Removing stale subst {letter}: → {dir_path}")
                subprocess.run(
                    ['subst', f'{letter}:', '/D'],
                    capture_output=True, check=False, timeout=5)
    except Exception as exc:
        log.debug(f"subst cleanup failed: {exc}")


def cleanup_stale_mounts(silent=False):
    """Detach orphaned VHDs, remove stale subst mappings, and delete
    leftover temp directories from previous sessions.

    Called automatically at startup (from ``MountManager.__init__``) and
    registered as an ``atexit`` handler.

    Parameters
    ----------
    silent : bool
        If True, suppress all log output (used by the atexit handler
        where the logging system may already be shut down).
    """
    if not is_windows():
        return

    cleaned = 0

    # ── VHD temp dirs ────────────────────────────────────────────────
    for vhd_dir in _find_stale_vhd_dirs():
        vhd_file = os.path.join(vhd_dir, 'pc98disk.vhd')

        if os.path.isfile(vhd_file) and _is_vhd_attached(vhd_file):
            if not silent:
                log.info(f"Detaching stale VHD: {vhd_file}")
            _detach_vhd_file(vhd_file)
            # Give Windows a moment to release handles.
            time.sleep(0.5)

        try:
            shutil.rmtree(vhd_dir, ignore_errors=True)
            cleaned += 1
            if not silent:
                log.info(f"Removed stale VHD dir: {vhd_dir}")
        except Exception as exc:
            if not silent:
                log.warning(f"Cannot remove {vhd_dir}: {exc}")

    # ── subst temp dirs ──────────────────────────────────────────────
    for subst_dir in _find_stale_subst_dirs():
        _remove_subst_for_dir(subst_dir)
        try:
            shutil.rmtree(subst_dir, ignore_errors=True)
            cleaned += 1
            if not silent:
                log.info(f"Removed stale subst dir: {subst_dir}")
        except Exception as exc:
            if not silent:
                log.warning(f"Cannot remove {subst_dir}: {exc}")

    if cleaned and not silent:
        log.info(f"Stale mount cleanup: removed {cleaned} leftover "
                 f"temp directories")


# =============================================================================
# Unified mount interface
# =============================================================================

class MountManager:
    """
    High-level mount interface.

    On Windows: tries VHD first, falls back to subst.
    On Linux/WSL: uses DirectoryMount.

    Mount identifiers:
      - Windows: drive letter string like "P"
      - Linux:   a short name or a full absolute path
    """

    STRATEGY_VHD = "vhd"
    STRATEGY_SUBST = "subst"
    STRATEGY_DIRECTORY = "directory"

    def __init__(self, mount_base=None):
        self._mounts = {}
        self._strategy = None
        self._mount_base = mount_base or DirectoryMount.default_base()

        # Clean up orphaned mounts from previous sessions that
        # weren't shut down properly (crash, killed process, etc.).
        try:
            cleanup_stale_mounts()
        except Exception as exc:
            log.warning(f"Stale mount cleanup failed: {exc}")

        # Safety net: if this process also exits unexpectedly, try
        # to unmount everything we created during this session.
        atexit.register(self._atexit_cleanup)

    @property
    def strategy(self):
        return self._strategy

    @property
    def mount_base(self):
        return self._mount_base

    @mount_base.setter
    def mount_base(self, value):
        self._mount_base = value

    def get_strategy_info(self):
        if self._strategy == self.STRATEGY_VHD:
            return (
                "Using VHD (Virtual Hard Disk).\n"
                "Explorer shows the correct image size.\n"
                "Files extracted from image; original is not modified."
            )
        elif self._strategy == self.STRATEGY_SUBST:
            return (
                "Using Windows `subst` (no admin).\n"
                "Explorer shows host drive's size (cosmetic only).\n"
                "Run as Administrator for correct size display via VHD."
            )
        elif self._strategy == self.STRATEGY_DIRECTORY:
            wsl_tag = " (WSL detected)" if _cached_is_wsl() else ""
            return (
                f"Using directory extraction{wsl_tag}.\n"
                f"Mount base: {self._mount_base}\n"
                "Original image is not modified."
            )
        return "No mount active."

    # -- key normalisation ------------------------------------------------

    def _mount_key(self, identifier):
        if is_windows():
            return identifier.upper().rstrip(':')
        else:
            if not os.path.isabs(identifier):
                return os.path.abspath(
                    os.path.join(self._mount_base, identifier))
            return os.path.abspath(identifier)

    # -- public API -------------------------------------------------------

    def mount(self, mount_id, mode, disk_image=None, fat_fs=None,
              image_size_bytes=0, prefer_vhd=True):
        key = self._mount_key(mount_id)
        if key in self._mounts:
            raise RuntimeError(f"{mount_id} is already mounted")
        if image_size_bytes == 0 and disk_image:
            image_size_bytes = (
                disk_image.total_sectors * disk_image.sector_size
            )
        if is_windows():
            return self._mount_windows(
                key, mode, disk_image, fat_fs,
                image_size_bytes, prefer_vhd
            )
        else:
            return self._mount_linux(
                key, mount_id, mode, disk_image, fat_fs)

    def unmount(self, mount_id):
        key = self._mount_key(mount_id)
        mount = self._mounts.pop(key, None)
        if mount:
            mount.unmount()
        return mount is not None

    def unmount_all(self):
        for key in list(self._mounts.keys()):
            mount = self._mounts.pop(key, None)
            if mount:
                mount.unmount()

    def _atexit_cleanup(self):
        """Last-resort cleanup registered via ``atexit``.

        Unmounts anything this session created that wasn't properly
        unmounted (e.g. if the user closed the window via the OS
        close button without clicking Unmount, or if the process was
        killed).  Also sweeps stale temp dirs one more time.
        """
        try:
            self.unmount_all()
        except Exception:
            pass
        try:
            cleanup_stale_mounts(silent=True)
        except Exception:
            pass

    def is_mounted(self, mount_id):
        key = self._mount_key(mount_id)
        mount = self._mounts.get(key)
        return mount is not None and mount.is_mounted

    def get_mount(self, mount_id):
        key = self._mount_key(mount_id)
        return self._mounts.get(key)

    def update(self, mount_id, mode, disk_image=None, fat_fs=None,
               save_path=None):
        """Write modifications from the mounted directory back into
        the disk image.

        For ``fat`` mode this rebuilds the entire FAT filesystem.
        For ``flat`` and ``sectors`` modes it copies the raw data back
        sector by sector.

        Returns a human-readable status string.
        """
        key = self._mount_key(mount_id)
        mount_obj = self._mounts.get(key)
        if mount_obj is None or not mount_obj.is_mounted:
            raise RuntimeError(f"{mount_id} is not mounted")

        # Resolve the real directory that backs the mount.
        mount_dir = self._resolve_content_dir(mount_obj)

        if mode == 'fat':
            if fat_fs is None:
                raise ValueError(
                    "FAT filesystem required for fat-mode update")
            files, dirs = fat_fs.write_back_from_directory(
                mount_dir, save_path=save_path)
            return (
                f"FAT write-back: {files} files, {dirs} directories "
                f"written to {'new file' if save_path else 'original'}"
            )
        elif mode == 'flat':
            if disk_image is None:
                raise ValueError(
                    "Disk image required for flat-mode update")
            n = _update_flat_from_dir(
                disk_image, mount_dir, save_path=save_path)
            return f"Flat write-back: {n} sectors written"
        elif mode == 'sectors':
            if disk_image is None:
                raise ValueError(
                    "Disk image required for sectors-mode update")
            n = _update_sectors_from_dir(
                disk_image, mount_dir, save_path=save_path)
            return f"Sector write-back: {n} sector files written"
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # -- internal ---------------------------------------------------------

    @staticmethod
    def _resolve_content_dir(mount_obj):
        """Return the real filesystem directory behind a mount object.

        For VHD and subst mounts the ``mount_point`` is a drive letter
        that might point to a VHD or subst alias; the real content
        directory may be different (e.g. ``_temp_dir`` for subst).  For
        DirectoryMount it's the same as mount_point.
        """
        if isinstance(mount_obj, VHDMount):
            # VHD: the drive letter *is* the content root.
            return mount_obj.mount_point + "\\"
        elif isinstance(mount_obj, SubstMount):
            # subst: the real directory is the temp dir.
            return mount_obj.content_dir or mount_obj.mount_point
        elif isinstance(mount_obj, DirectoryMount):
            return mount_obj.mount_point
        # Fallback.
        return mount_obj.mount_point

    def _mount_windows(self, drive, mode, disk_image, fat_fs,
                       image_size_bytes, prefer_vhd):
        if prefer_vhd:
            try:
                mount = self._try_vhd_mount(
                    drive, mode, disk_image, fat_fs, image_size_bytes)
                self._mounts[drive] = mount
                self._strategy = self.STRATEGY_VHD
                return mount
            except Exception as e:
                log.warning(
                    f"VHD mount failed, falling back to subst: {e}")
        mount = SubstMount(drive)
        self._do_mount(mount, mode, disk_image, fat_fs)
        self._mounts[drive] = mount
        self._strategy = self.STRATEGY_SUBST
        return mount

    def _mount_linux(self, key, mount_id, mode, disk_image, fat_fs):
        if not os.path.isabs(mount_id):
            mount_path = os.path.join(self._mount_base, mount_id)
        else:
            mount_path = mount_id
        mount = DirectoryMount(mount_path)
        self._do_mount(mount, mode, disk_image, fat_fs)
        self._mounts[key] = mount
        self._strategy = self.STRATEGY_DIRECTORY
        return mount

    @staticmethod
    def _do_mount(mount, mode, disk_image, fat_fs):
        if mode == 'fat':
            if fat_fs is None:
                raise ValueError(
                    "FAT filesystem required for fat mode")
            mount.mount_fat(fat_fs)
        elif mode == 'flat':
            if disk_image is None:
                raise ValueError(
                    "Disk image required for flat mode")
            mount.mount_flat(disk_image)
        elif mode == 'sectors':
            if disk_image is None:
                raise ValueError(
                    "Disk image required for sectors mode")
            mount.mount_sectors(disk_image)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _try_vhd_mount(self, drive, mode, disk_image, fat_fs,
                       image_size_bytes):
        mount = VHDMount(drive)
        if mode == 'fat':
            if fat_fs is None:
                raise ValueError("FAT filesystem required")
            mount.mount_fat(fat_fs, image_size_bytes)
        elif mode == 'flat':
            if disk_image is None:
                raise ValueError("Disk image required")
            mount.mount_flat(disk_image)
        elif mode == 'sectors':
            if disk_image is None:
                raise ValueError("Disk image required")
            mount.mount_sectors(disk_image)
        return mount