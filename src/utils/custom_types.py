"""
Common typing definitions and TypedDict structures for evaluation and pipeline modules.
"""

from typing import List, Tuple, TypedDict


class BenchmarkConfig(TypedDict):
    """Specification of an automated comparison benchmark target."""

    module: str
    yaml_key: str
    yaml_file: str
    report_prefix: str
    extra_args: List[Tuple[str, str]]
