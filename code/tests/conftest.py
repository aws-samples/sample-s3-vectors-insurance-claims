# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Shared test helpers.

Every Lambda function lives in its own directory with an ``index.py`` entry
point. Putting all of those directories on ``sys.path`` and running
``import index`` is ambiguous - Python returns whichever directory comes
first - so tests load each handler under a unique module name instead:

    search_index = load_lambda("search_handler")
    @patch("search_handler_index.s3vectors_client")

The module is registered in ``sys.modules`` under ``<dir>_index`` so that
``unittest.mock.patch`` can target it by name.
"""

import importlib.util
import os
import sys

# boto3 clients are created at import time; give them a Region so tests run
# without AWS configuration. No AWS calls are made - clients are mocked.
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

LAMBDA_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lambda")
)


def load_lambda(function_dir: str):
    """Import ``lambda/<function_dir>/index.py`` as ``<function_dir>_index``.

    Set any environment variables the handler reads at import time
    *before* calling this.
    """
    module_name = f"{function_dir}_index"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = os.path.join(LAMBDA_ROOT, function_dir, "index.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
