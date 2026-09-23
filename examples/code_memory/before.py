def retry_limit():
    return 3


def should_retry(attempt):
    return attempt < retry_limit()
