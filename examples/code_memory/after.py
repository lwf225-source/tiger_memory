def retry_limit():
    return 1


def should_retry(attempt):
    return attempt < retry_limit()
