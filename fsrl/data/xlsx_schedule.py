"""Dependency-free reader for the simple tabular XLSX experiment schedules."""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
_COLUMN = re.compile(r"([A-Z]+)")


def _column_index(reference: str) -> int:
    match = _COLUMN.match(reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(path))
    values = []
    for item in root.findall(f"{{{_MAIN}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN}}}t")))
    return values


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f"{{{_MAIN}}}sheets/{{{_MAIN}}}sheet")
    if sheet is None:
        raise ValueError("XLSX workbook has no sheets")
    relationship_id = sheet.attrib[f"{{{_REL}}}id"]
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{{{_PACKAGE_REL}}}Relationship"):
        if relationship.attrib["Id"] == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            if not target.startswith("xl/"):
                target = posixpath.normpath(posixpath.join("xl", target))
            return target
    raise ValueError("first worksheet relationship is missing")


def _cell_value(cell: ET.Element, shared: list[str]):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN}}}t"))
    node = cell.find(f"{{{_MAIN}}}v")
    if node is None or node.text is None:
        return None
    text = node.text
    if cell_type == "s":
        return shared[int(text)]
    if cell_type == "b":
        return bool(int(text))
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def read_first_sheet_records(path: str | Path) -> list[dict[str, object]]:
    """Read the first worksheet as header-keyed records.

    This intentionally supports the small subset used by the public design
    schedules: shared/inline strings, booleans, and cached numeric cells.
    """

    with zipfile.ZipFile(Path(path)) as archive:
        shared = _shared_strings(archive)
        root = ET.fromstring(archive.read(_first_sheet_path(archive)))
    rows = []
    for row in root.findall(f".//{{{_MAIN}}}sheetData/{{{_MAIN}}}row"):
        values: dict[int, object] = {}
        for cell in row.findall(f"{{{_MAIN}}}c"):
            values[_column_index(cell.attrib["r"])] = _cell_value(cell, shared)
        if values:
            rows.append(values)
    if not rows:
        return []
    width = max(max(row) for row in rows) + 1
    headers = [str(rows[0].get(index, "")).strip() for index in range(width)]
    if not all(headers):
        raise ValueError("XLSX schedule contains an empty header")
    records = []
    for row in rows[1:]:
        record = {headers[index]: row.get(index) for index in range(width)}
        if any(value is not None for value in record.values()):
            records.append(record)
    return records
