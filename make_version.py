"""Writes version_info.txt (Windows file properties for the exe) from VERSION
in ScroomMacro.pyw. Used by build_exe.bat. Prints the version."""
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ScroomMacro.pyw"), encoding="utf-8").read()
ver = re.search(r'^VERSION = "([\d.]+)"', src, re.M).group(1)
parts = [int(p) for p in ver.split(".")] + [0, 0, 0]
t = tuple(parts[:4])

info = f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={t}, prodvers={t}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Scroom Macro'),
      StringStruct('FileDescription', 'Scroom Macro - fishing minigame helper for Roblox'),
      StringStruct('FileVersion', '{ver}'),
      StringStruct('InternalName', 'ScroomMacro'),
      StringStruct('OriginalFilename', 'ScroomMacro.exe'),
      StringStruct('ProductName', 'Scroom Macro'),
      StringStruct('ProductVersion', '{ver}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
with open(os.path.join(here, "version_info.txt"), "w", encoding="utf-8") as f:
    f.write(info)
print(ver)
