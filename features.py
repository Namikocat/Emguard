"""Feature extraction for Emguard.

The model is trained on EMBER2024 (https://github.com/FutureComputing4AI/EMBER2024),
whose records hold "EMBER feature version 3" raw features computed with pefile.
This module is an independent implementation of the parts of that format the
model uses, so training and the app compute features the same way:

1. ``raw_features(bytez)`` computes, from the bytes of a file, the raw-feature
   fields used here, in the same layout as the EMBER2024 ``*.jsonl`` records
   (following the definitions in thrember's features.py, Apache-2.0).
2. ``vectorize(raw)`` turns such a record (from a file or from the dataset)
   into the model's input vector.

Only properties of the file itself are used, never dataset metadata.
"""

import math
import re
from collections import Counter

import numpy as np
import pefile

_ALL_STRINGS = re.compile(rb"[\x20-\x7f]{5,}")


# ---------------------------------------------------------------------------
# Raw features (EMBER v3 layout)
# ---------------------------------------------------------------------------

def parse_pe(bytez):
    """Parse bytes as a PE file with pefile. Returns pefile.PE, or None if invalid."""
    try:
        return pefile.PE(data=bytez)
    except (pefile.PEFormatError, AttributeError):
        return None
    except Exception:
        return None


def _entropy(data):
    if not data:
        return 0.0
    n = len(data)
    return -sum(c / n * math.log(c / n, 2) for c in Counter(data).values())


def _byte_entropy_histogram(a, step=1024, window=2048):
    def entropy_bin_counts(block):
        c = np.bincount(block >> 4, minlength=16)
        p = c.astype(np.float32) / window
        wh = np.where(c)[0]
        h = np.sum(-p[wh] * np.log2(p[wh])) * 2
        hbin = int(h * 2)
        if hbin == 16:
            hbin = 15
        return hbin, c

    output = np.zeros((16, 16), dtype=np.int64)
    if a.shape[0] < window:
        hbin, c = entropy_bin_counts(a)
        output[hbin, :] += c
    else:
        for block in np.lib.stride_tricks.sliding_window_view(a, window)[::step, :]:
            hbin, c = entropy_bin_counts(block)
            output[hbin, :] += c
    return output.flatten().tolist()


def _strings(bytez):
    allstrings = _ALL_STRINGS.findall(bytez)
    if allstrings:
        avlength = sum(len(s) for s in allstrings) / len(allstrings)
        c = np.bincount(np.frombuffer(b"".join(allstrings), dtype=np.uint8) - 0x20, minlength=96)
        csum = int(c.sum())
        p = c.astype(np.float32) / csum
        wh = np.where(c)[0]
        entropy = float(np.sum(-p[wh] * np.log2(p[wh])))
    else:
        avlength, c, csum, entropy = 0, np.zeros(96, dtype=np.int64), 0, 0.0
    return {"numstrings": len(allstrings), "avlength": avlength,
            "printabledist": c.tolist(), "printables": csum, "entropy": entropy}


def _header(pe):
    fh, oh = pe.FILE_HEADER, pe.OPTIONAL_HEADER
    return {
        "coff": {
            "number_of_sections": fh.NumberOfSections,
            "number_of_symbols": fh.NumberOfSymbols,
            "characteristics": [k[11:] for k, v in fh.__dict__.items() if k.startswith("IMAGE_FILE_") and v],
        },
        "optional": {
            "magic": oh.Magic,
            "subsystem": pefile.SUBSYSTEM_TYPE.get(oh.Subsystem, "IMAGE_SUBSYSTEM_UNKNOWN"),
            "major_linker_version": oh.MajorLinkerVersion,
            "minor_linker_version": oh.MinorLinkerVersion,
            "major_operating_system_version": oh.MajorOperatingSystemVersion,
            "major_subsystem_version": oh.MajorSubsystemVersion,
            "sizeof_code": oh.SizeOfCode,
            "sizeof_headers": oh.SizeOfHeaders,
            "sizeof_image": oh.SizeOfImage,
            "sizeof_initialized_data": oh.SizeOfInitializedData,
            "sizeof_uninitialized_data": oh.SizeOfUninitializedData,
            "sizeof_stack_reserve": oh.SizeOfStackReserve,
            "sizeof_heap_reserve": oh.SizeOfHeapReserve,
            "checksum": oh.CheckSum,
            "dll_characteristics": [
                k[25:] for k, v in oh.__dict__.items() if k.startswith("IMAGE_DLLCHARACTERISTICS_") and v
            ],
        },
    }


def _section(bytez, pe):
    def name(s):
        return s.Name.strip(b"\x00").decode(errors="ignore").lower()

    entry = ""
    aoep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    for s in pe.sections:
        if s.contains_rva(aoep):
            entry = name(s)
    i = 0
    while entry == "" and i < len(pe.sections):
        if pe.sections[i].Characteristics & 0x20000000 > 0:
            entry = name(pe.sections[i])
        i += 1

    sections = [{
        "name": name(s),
        "size": s.SizeOfRawData,
        "entropy": s.get_entropy(),
        "vsize": s.Misc_VirtualSize,
        "props": [sc[10:] for sc, _ in pefile.section_characteristics if s.__dict__[sc]],
    } for s in pe.sections]

    overlay = {"size": 0, "size_ratio": 0, "entropy": 0}
    data = pe.get_overlay()
    if data is not None:
        overlay = {"size": len(data), "size_ratio": len(data) / len(bytez), "entropy": _entropy(data)}
    return {"entry": entry, "sections": sections, "overlay": overlay}


def _imports(pe):
    imports = {}
    for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
        dll = entry.dll.decode(errors="ignore")
        imports[dll] = []
        for lib in entry.imports:
            if lib.name is not None and len(lib.name):
                imports[dll].append(lib.name.decode(errors="ignore"))
            elif lib.ordinal is not None:
                imports[dll].append(f"{dll}:ordinal{lib.ordinal}")
    return imports


def _exports(pe):
    if not hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        return []
    return [e.name.decode(errors="ignore") if e.name else f"ordinal{e.ordinal}"
            for e in pe.DIRECTORY_ENTRY_EXPORT.symbols]


def _datadirectories(pe):
    out = [{"has_relocs": int(pe.has_relocs()), "has_dynamic_relocs": int(pe.has_dynamic_relocs())}]
    for d in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
        out.append({"name": str(d.name).replace("IMAGE_DIRECTORY_ENTRY_", ""),
                    "size": d.Size, "virtual_address": d.VirtualAddress})
    return out


def raw_features(bytez, pe=None):
    """EMBER v3-format raw features (the fields used by vectorize) for file bytes."""
    if pe is None:
        pe = parse_pe(bytez)
    a = np.frombuffer(bytez, dtype=np.uint8)
    raw = {
        "histogram": np.bincount(a, minlength=256).tolist(),
        "byteentropy": _byte_entropy_histogram(a),
        "strings": _strings(bytez),
        "general": {"size": len(bytez), "entropy": _entropy(bytez), "is_pe": int(pe is not None)},
        "header": {}, "section": {}, "imports": {}, "exports": [],
        "datadirectories": [], "pefilewarnings": [],
    }
    if pe is not None:
        raw.update({
            "header": _header(pe),
            "section": _section(bytez, pe),
            "imports": _imports(pe),
            "exports": _exports(pe),
            "datadirectories": _datadirectories(pe),
            "pefilewarnings": list(set(pe.get_warnings())),
        })
    return raw


# ---------------------------------------------------------------------------
# Vectorization
# ---------------------------------------------------------------------------

SUBSYSTEMS = ["IMAGE_SUBSYSTEM_WINDOWS_GUI", "IMAGE_SUBSYSTEM_WINDOWS_CUI", "IMAGE_SUBSYSTEM_NATIVE"]
COFF_FLAGS = ["DLL", "EXECUTABLE_IMAGE", "LARGE_ADDRESS_AWARE", "RELOCS_STRIPPED", "32BIT_MACHINE",
              "DEBUG_STRIPPED", "LINE_NUMS_STRIPPED", "LOCAL_SYMS_STRIPPED"]
DLL_FLAGS = ["HIGH_ENTROPY_VA", "DYNAMIC_BASE", "FORCE_INTEGRITY", "NX_COMPAT", "NO_ISOLATION",
             "NO_SEH", "NO_BIND", "APPCONTAINER", "WDM_DRIVER", "GUARD_CF", "TERMINAL_SERVER_AWARE"]
OPTIONAL_SIZES = ["sizeof_code", "sizeof_headers", "sizeof_image", "sizeof_initialized_data",
                  "sizeof_uninitialized_data", "sizeof_stack_reserve", "sizeof_heap_reserve"]
OPTIONAL_VERSIONS = ["major_linker_version", "minor_linker_version",
                     "major_operating_system_version", "major_subsystem_version"]
DATA_DIRECTORIES = ["EXPORT", "IMPORT", "RESOURCE", "EXCEPTION", "SECURITY", "BASERELOC", "DEBUG",
                    "COPYRIGHT", "GLOBALPTR", "TLS", "LOAD_CONFIG", "BOUND_IMPORT", "IAT",
                    "DELAY_IMPORT", "COM_DESCRIPTOR"]
STANDARD_SECTIONS = {".text", ".data", ".rdata", ".rsrc", ".reloc", ".pdata", ".idata", ".edata",
                     ".tls", ".bss", ".crt", ".didat", ".gfids", ".00cfg"}
IMPORT_DLLS = ["kernel32.dll", "user32.dll", "advapi32.dll", "shell32.dll", "ole32.dll", "oleaut32.dll",
               "ws2_32.dll", "wsock32.dll", "wininet.dll", "winhttp.dll", "urlmon.dll", "crypt32.dll",
               "ntdll.dll", "mscoree.dll", "msvbvm60.dll", "gdi32.dll", "comctl32.dll", "shlwapi.dll",
               "psapi.dll", "iphlpapi.dll", "netapi32.dll", "msvcrt.dll", "version.dll", "bcrypt.dll"]
# Commonly abused Windows APIs (A/W suffixes are merged).
IMPORT_APIS = [
    "VirtualAlloc", "VirtualAllocEx", "VirtualProtect", "VirtualProtectEx", "WriteProcessMemory",
    "ReadProcessMemory", "CreateRemoteThread", "NtUnmapViewOfSection", "ResumeThread", "SetThreadContext",
    "GetThreadContext", "OpenProcess", "CreateProcess", "WinExec", "ShellExecute", "ShellExecuteEx",
    "LoadLibrary", "LoadLibraryEx", "GetProcAddress", "GetModuleHandle", "IsDebuggerPresent",
    "CheckRemoteDebuggerPresent", "OutputDebugString", "GetTickCount", "QueryPerformanceCounter",
    "Sleep", "SetWindowsHookEx", "GetAsyncKeyState", "GetKeyState", "GetForegroundWindow",
    "FindWindow", "RegOpenKeyEx", "RegSetValueEx", "RegCreateKeyEx", "RegDeleteKey",
    "CreateService", "StartService", "OpenSCManager", "AdjustTokenPrivileges", "OpenProcessToken",
    "LookupPrivilegeValue", "CryptEncrypt", "CryptDecrypt", "CryptAcquireContext", "CryptGenKey",
    "InternetOpen", "InternetOpenUrl", "InternetReadFile", "HttpSendRequest", "URLDownloadToFile",
    "WSAStartup", "socket", "connect", "send", "recv", "gethostbyname", "CreateToolhelp32Snapshot",
    "Process32First", "Process32Next", "CreateMutex", "OpenMutex", "FindFirstFile", "FindNextFile",
    "DeleteFile", "MoveFileEx", "CopyFile", "CreateFile", "WriteFile", "SetFileAttributes",
    "GetTempPath", "GetSystemDirectory", "GetComputerName", "GetUserName", "SetErrorMode",
    "FindResource", "LoadResource", "LockResource", "SizeofResource", "TerminateProcess",
    "ExitProcess", "CreateThread", "SuspendThread", "QueueUserAPC", "NtQueryInformationProcess",
    "RtlDecompressBuffer", "MapViewOfFile", "CreateFileMapping", "OpenClipboard", "GetClipboardData",
    "BitBlt", "GetDC", "keybd_event", "mouse_event", "SystemParametersInfo", "_CorExeMain", "_CorDllMain",
]
_API_INDEX = {name: i for i, name in enumerate(IMPORT_APIS)}


def _api_base(name):
    if len(name) > 2 and name[-1] in "AW" and name[-2].islower():
        return name[:-1]
    return name


def _log(x):
    return math.log1p(max(float(x), 0.0))


def _normalize(counts):
    counts = np.asarray(counts, dtype=np.float32)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _header_vector(h):
    n = 4 + len(COFF_FLAGS) + len(SUBSYSTEMS) + 1 + len(OPTIONAL_VERSIONS) + len(OPTIONAL_SIZES) + 1 + len(DLL_FLAGS)
    if not h:
        return [0.0] * n
    coff, opt = h["coff"], h["optional"]
    chars, dllc = set(coff["characteristics"]), set(opt["dll_characteristics"])
    return (
        [1.0, float(opt["magic"] == 0x20B), coff["number_of_sections"], _log(coff["number_of_symbols"])]
        + [float(f in chars) for f in COFF_FLAGS]
        + [float(opt["subsystem"] == s) for s in SUBSYSTEMS]
        + [float(opt["subsystem"] not in SUBSYSTEMS)]
        + [float(opt[k]) for k in OPTIONAL_VERSIONS]
        + [_log(opt[k]) for k in OPTIONAL_SIZES]
        + [float(opt["checksum"] == 0)]
        + [float(f in dllc) for f in DLL_FLAGS]
    )


def _section_vector(sec):
    n = 16
    if not sec or "sections" not in sec:
        return [0.0] * n
    ss = sec["sections"]
    ent = [s["entropy"] for s in ss] or [0.0]
    entry = next((s for s in ss if s["name"] == sec["entry"]), None)
    wx = sum(1 for s in ss if "MEM_WRITE" in s["props"] and "MEM_EXECUTE" in s["props"])
    ov = sec.get("overlay") or {"size": 0, "size_ratio": 0, "entropy": 0}
    return [
        len(ss), max(ent), float(np.mean(ent)), min(ent),
        float(entry["entropy"]) if entry else 0.0,
        float(sec["entry"] == ".text"), float(sec["entry"] == ""),
        float(wx), float(sum(1 for s in ss if s["size"] == 0)),
        float(sum(1 for s in ss if s["name"] not in STANDARD_SECTIONS)),
        float(sum(1 for s in ss if s["entropy"] > 7.0)),
        float(max((s["vsize"] / max(s["size"], 1) for s in ss), default=0.0) > 10),
        _log(ov["size"]), float(ov["size_ratio"]), float(ov["entropy"]),
        float(any(s["name"] in (".upx0", "upx0", "upx1", ".upx1", ".aspack", ".themida") for s in ss)),
    ]


def _imports_vector(imports):
    dlls = {d.lower() for d in imports}
    funcs = [f for fs in imports.values() for f in fs]
    api = np.zeros(len(IMPORT_APIS), dtype=np.float32)
    for f in funcs:
        i = _API_INDEX.get(_api_base(f))
        if i is not None:
            api[i] = 1.0
    return (
        [_log(len(imports)), _log(len(funcs)), _log(sum(1 for f in funcs if ":ordinal" in f))]
        + [float(d in dlls) for d in IMPORT_DLLS]
        + api.tolist()
    )


def _datadir_vector(dd):
    if not dd:
        return [0.0] * (2 + len(DATA_DIRECTORIES))
    sizes = {d["name"]: d["size"] for d in dd[1:]}
    return [float(dd[0]["has_relocs"]), float(dd[0]["has_dynamic_relocs"])] + [
        _log(sizes.get(n, 0)) for n in DATA_DIRECTORIES
    ]


def vectorize(raw):
    """Turn an EMBER v3 raw feature record into the model input vector."""
    s, g = raw["strings"], raw["general"]
    parts = [
        _normalize(raw["histogram"]),
        _normalize(raw["byteentropy"]),
        [_log(s["numstrings"]), _log(s["avlength"]), _log(s["printables"]), s["entropy"]],
        _normalize(s["printabledist"]),
        [_log(g["size"]), g["entropy"], float(g["is_pe"])],
        _header_vector(raw.get("header")),
        _section_vector(raw.get("section")),
        _imports_vector(raw.get("imports") or {}),
        [_log(len(raw.get("exports") or []))],
        _datadir_vector(raw.get("datadirectories")),
        [float(len(raw.get("pefilewarnings") or []) > 0)],
    ]
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])


N_FEATURES = len(vectorize(raw_features(b"MZ" + b"\x00" * 64)))


def summary(raw):
    """Small human-readable summary of a file's features for display."""
    h = raw.get("header") or {}
    sec = raw.get("section") or {}
    return {
        "size": raw["general"]["size"],
        "entropy": round(raw["general"]["entropy"], 3),
        "is_64bit": bool(h and h["optional"]["magic"] == 0x20B),
        "is_dll": bool(h and "DLL" in h["coff"]["characteristics"]),
        "sections": [s["name"] for s in sec.get("sections", [])],
        "entry_section": sec.get("entry"),
        "imported_dlls": sorted(raw.get("imports", {})),
        "num_imported_functions": sum(len(v) for v in raw.get("imports", {}).values()),
        "num_exports": len(raw.get("exports", [])),
        "num_strings": raw["strings"]["numstrings"],
        "pefile_warnings": len(raw.get("pefilewarnings", [])),
    }
