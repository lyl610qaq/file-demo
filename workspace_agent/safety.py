import os
import stat
from pathlib import Path


class PathRejected(ValueError):
    pass


def _is_link_or_reparse(path: Path) -> bool:
    metadata = os.lstat(path)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        file_attributes & reparse_flag
    )


class WorkspaceGuard:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        expanded = Path(root).expanduser()
        absolute = Path(os.path.abspath(expanded))

        if os.path.lexists(absolute) and _is_link_or_reparse(absolute):
            raise PathRejected("workspace root is not a physical directory")

        absolute.mkdir(parents=True, exist_ok=True)

        if _is_link_or_reparse(absolute):
            raise PathRejected("workspace root is not a physical directory")

        self.root = Path(os.path.realpath(absolute))

    def resolve(self, relative: str, must_exist: bool = False) -> Path:
        if not isinstance(relative, str) or not relative:
            raise PathRejected("path must be a non-empty relative string")
        if "\x00" in relative:
            raise PathRejected("path contains an invalid character")

        normalized = relative.replace("\\", os.sep).replace("/", os.sep)
        drive, _ = os.path.splitdrive(normalized)
        if drive or os.path.isabs(normalized):
            raise PathRejected("absolute paths are not allowed")

        components = normalized.split(os.sep)
        if ".." in components:
            raise PathRejected("parent path segments are not allowed")

        lexical = self.root
        for component in components:
            if component in ("", "."):
                continue
            lexical = lexical / component
            if os.path.lexists(lexical) and _is_link_or_reparse(lexical):
                raise PathRejected("path contains a non-physical component")

        candidate = Path(os.path.realpath(lexical))
        try:
            common = os.path.commonpath((self.root, candidate))
        except ValueError as exc:
            raise PathRejected("path is outside the workspace") from exc
        if Path(common) != self.root:
            raise PathRejected("path is outside the workspace")

        if must_exist and not candidate.exists():
            raise FileNotFoundError("workspace path does not exist")

        return candidate
