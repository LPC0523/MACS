# tools/packaging

    python tools/packaging/check_package.py          # audit a built dist/MACS
    python tools/packaging/test_packaged_shutdown.py # exe closes with code 0

`check_package.py` runs after `pyinstaller macs.spec` and answers what a
successful build does not: whether anything that should never ship got in, and
whether the version was really stamped.
