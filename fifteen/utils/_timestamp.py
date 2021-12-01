import datetime


def timestamp() -> str:
    """Get a current timestamp as a string. Example format: `2021-11-05-15:46:32`."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
