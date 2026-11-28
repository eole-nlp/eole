"""Shared quantization selection using legacy names or model-root paths."""

from fnmatch import fnmatchcase


def matches_module(path, patterns):
    """Bare names match any component; dotted patterns match the full path.

    Globs use shell-style matching (``*`` can span multiple path components).
    """
    return any(
        fnmatchcase(path, pattern) if "." in pattern else any(fnmatchcase(part, pattern) for part in path.split("."))
        for pattern in patterns
    )


def is_excluded(path, patterns):
    """An excluded parent excludes its entire subtree."""
    parts = path.split(".")
    return any(matches_module(".".join(parts[:end]), patterns) for end in range(1, len(parts) + 1))


def is_selected(path, includes, excludes=()):
    # Legacy includes select leaf names, whereas legacy exclusions select parents too.
    leaf = path.rsplit(".", 1)[-1]
    included = any(fnmatchcase(path if "." in pattern else leaf, pattern) for pattern in includes)
    return included and not is_excluded(path, excludes)
