import List
import Any
import time
import traceback


def wait_for_condition(
    condition_predictor,
    timeout=10,
    retry_interval_ms=100,
    raise_exceptions=False,
    **kwargs: Any,
):
    """Wait until a condition is met or time out with an exception.

    Args:
        condition_predictor: A function that predicts the condition.
        timeout: Maximum timeout in seconds.
        retry_interval_ms: Retry interval in milliseconds.
        raise_exceptions: If true, exceptions that occur while executing
            condition_predictor won't be caught and instead will be raised.

    Raises:
        RuntimeError: If the condition is not met before the timeout expires.
    """
    start = time.time()
    last_ex = None
    while time.time() - start <= timeout:
        try:
            if condition_predictor(**kwargs):
                return
        except Exception:
            if raise_exceptions:
                raise
            last_ex = traceback.format_exc()
        time.sleep(retry_interval_ms / 1000.0)
    message = "The condition wasn't met before the timeout expired."
    if last_ex is not None:
        message += f" Last exception: {last_ex}"
    raise RuntimeError(message)
