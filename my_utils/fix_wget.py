import wget


def detect_filename(url=None, out=None, headers=None, default="download.wget"):
    """Return filename for saving file. If no filename is detected from output
    argument, url or headers, return default (download.wget)
    """
    name = default
    if out:
        name = out
    elif url:
        name = wget.filename_from_url(url)
    elif headers:
        name = wget.filename_from_headers(headers)
    return name


def fix_wget():
    wget.detect_filename = detect_filename
