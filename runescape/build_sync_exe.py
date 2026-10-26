"""Make .venv/Scripts/RS GE Sync.exe, so the sync shows up by name (and with its own icon) in Task Manager.

It's a renamed copy of this Python's pythonw.exe (plus the DLLs it loads), not a frozen app: it runs the live
sync_cache.py with the venv's packages (Python finds .venv/pyvenv.cfg next to Scripts/), so code changes need no
rebuild. Only the copy's icon and version info (the name Task Manager shows) are changed. Re-run after the
venv is recreated or uv upgrades Python; install_sync_task.ps1 does that.

    uv run build_sync_exe.py
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import sys
from ctypes import wintypes
from pathlib import Path

from sync_tray import APP_NAME, icon_image

SCRIPTS_DIR = Path(sys.prefix) / "Scripts"
EXE_PATH = SCRIPTS_DIR / f"{APP_NAME}.exe"
ICON_PATH = SCRIPTS_DIR / "rs-ge-sync.ico"
RUNTIME_FILES = ("python3.dll", f"python{sys.version_info.major}{sys.version_info.minor}.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll")
ICON_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

RT_ICON, RT_GROUP_ICON, RT_VERSION = 3, 14, 16
LANG_EN_US = 0x0409  # the language of pythonw.exe's resources

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
kernel32.UpdateResourceW.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, wintypes.WORD,
                                     ctypes.c_void_p, wintypes.DWORD]
kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
kernel32.LoadLibraryExW.restype = wintypes.HMODULE
kernel32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]
ENUM_NAMES_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_void_p)
kernel32.EnumResourceNamesW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ENUM_NAMES_PROC, ctypes.c_void_p]


def check(ok) -> None:
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())


def resource_ids(path: Path, res_type: int) -> list[int]:
    module = kernel32.LoadLibraryExW(str(path), None, 0x22)  # LOAD_LIBRARY_AS_DATAFILE | AS_IMAGE_RESOURCE
    check(module)
    ids = []
    callback = ENUM_NAMES_PROC(lambda module, res_type, name, param: ids.append(name) or True)
    kernel32.EnumResourceNamesW(module, res_type, callback, None)
    kernel32.FreeLibrary(module)
    return ids


def version_node(key: str, value: bytes = b"", children: bytes = b"", text: bool = False,
                 binary: bool = False) -> bytes:
    """One VS_VERSIONINFO block (wLength, wValueLength, wType, key, value, children), padded to 4 bytes."""
    key_bytes = (key + "\0").encode("utf-16-le")
    header = 6 + len(key_bytes)
    header += -header % 4
    body = value + b"\0" * (-len(value) % 4) + children
    value_length = len(value) // 2 if text else len(value)
    node = struct.pack("<HHH", header + len(body), value_length, 0 if binary else 1) + key_bytes
    return node + b"\0" * (header - 6 - len(key_bytes)) + body


def version_info(description: str) -> bytes:
    fixed = struct.pack("<13I", 0xFEEF04BD, 0x10000, 0x10000, 0, 0x10000, 0, 0x3F, 0,
                        0x40004, 1, 0, 0, 0)  # VOS_NT_WINDOWS32, VFT_APP, version 1.0
    strings = {"FileDescription": description, "ProductName": description, "FileVersion": "1.0",
               "ProductVersion": "1.0", "InternalName": "rs-ge-sync", "OriginalFilename": EXE_PATH.name,
               "Comments": f"Renamed Python {sys.version.split()[0]} interpreter running sync_cache.py"}
    table = b"".join(version_node(k, (v + "\0").encode("utf-16-le"), text=True) for k, v in strings.items())
    string_info = version_node("StringFileInfo", children=version_node("040904B0", children=table))
    var_info = version_node("VarFileInfo", children=version_node(
        "Translation", struct.pack("<HH", LANG_EN_US, 1200), binary=True))
    return version_node("VS_VERSION_INFO", fixed, string_info + var_info, binary=True)


def stamp(exe: Path, ico: Path, description: str) -> None:
    """Replace the exe's icons with the .ico's and its version info with one naming `description`."""
    old_icons, old_groups = resource_ids(exe, RT_ICON), resource_ids(exe, RT_GROUP_ICON)
    ico_data = ico.read_bytes()
    count = struct.unpack_from("<HHH", ico_data)[2]
    handle = kernel32.BeginUpdateResourceW(str(exe), False)
    check(handle)
    buffers = []  # keep the data alive until EndUpdateResource writes it

    def update(res_type: int, res_id: int, data: bytes | None) -> None:
        buf = ctypes.create_string_buffer(data, len(data)) if data is not None else None
        buffers.append(buf)
        check(kernel32.UpdateResourceW(handle, res_type, res_id, LANG_EN_US, buf, len(data) if data else 0))

    for res_id in old_icons:
        update(RT_ICON, res_id, None)
    group = struct.pack("<HHH", 0, 1, count)
    for i in range(count):
        width, height, colors, reserved, planes, bits, size, offset = struct.unpack_from("<BBBBHHII", ico_data,
                                                                                          6 + 16 * i)
        update(RT_ICON, i + 1, ico_data[offset:offset + size])
        group += struct.pack("<BBBBHHIH", width, height, colors, reserved, planes, bits, size, i + 1)
    for res_id in old_groups:
        update(RT_GROUP_ICON, res_id, None)
    update(RT_GROUP_ICON, 1, group)
    update(RT_VERSION, 1, version_info(description))
    check(kernel32.EndUpdateResourceW(handle, False))


def main() -> None:
    if sys.platform != "win32" or sys.prefix == sys.base_prefix:
        raise SystemExit("Run this with the project's venv on Windows: uv run build_sync_exe.py")
    base = Path(sys.base_prefix)
    for name in RUNTIME_FILES:
        shutil.copy2(base / name, SCRIPTS_DIR / name)
    shutil.copy2(base / "pythonw.exe", EXE_PATH)
    icon_image().save(ICON_PATH, sizes=ICON_SIZES)
    stamp(EXE_PATH, ICON_PATH, APP_NAME)
    print(f"Built {EXE_PATH}")


if __name__ == "__main__":
    main()
