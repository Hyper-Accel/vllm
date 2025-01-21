import warnings

try:
    import vllm.commit_id
    __commit__ = vllm.commit_id.__commit__
except Exception as e:
    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)
    __commit__ = "COMMIT_HASH_PLACEHOLDER"

__version__ = "0.6.1"
__version_tuple__ = (0, 6, 1)
'''
try:
    from ._version import __version__, __version_tuple__
except Exception as e:
    import warnings

    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
'''
