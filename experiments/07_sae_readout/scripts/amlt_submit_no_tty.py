"""Submit AMLT yamls without a controlling /dev/tty.

amlt v11.4+ uses click.getchar which opens /dev/tty directly for confirmation
prompts; that bypasses stdin so `yes y |` doesn't help in non-tty shells.
This wrapper monkey-patches click.getchar (and amlt.helpers.console.getchar)
to return 'y' immediately, then delegates to the standard amlt CLI.

Usage:
    python amlt_submit_no_tty.py YAML_PATH "Description text"
or:
    python amlt_submit_no_tty.py YAML_PATH JOB_NAME "Description text"
"""
from __future__ import annotations
import os, sys


def patch_console():
    """Replace click.getchar + amlt console.fast_confirm with always-yes stubs."""
    try:
        import click
        click.getchar = lambda echo=False: 'y'
    except ImportError:
        print('[warn] click not importable; skipping patch', file=sys.stderr)

    try:
        from amlt.helpers import console as amlt_console
        amlt_console.getchar = lambda echo=False: 'y'
        if hasattr(amlt_console, 'fast_confirm'):
            amlt_console.fast_confirm = lambda *a, **k: True
        if hasattr(amlt_console, 'confirm'):
            amlt_console.confirm = lambda *a, **k: True
    except ImportError:
        print('[warn] amlt.helpers.console not importable; skipping patch',
              file=sys.stderr)

    try:
        from rich import prompt as rich_prompt
        # rich.prompt.Confirm uses .ask which prompts on stdin
        original_ask = getattr(rich_prompt.Confirm, 'ask', None)
        if original_ask is not None:
            rich_prompt.Confirm.ask = classmethod(lambda cls, *a, **k: True)
    except ImportError:
        pass


def main():
    if len(sys.argv) < 3:
        print('Usage: amlt_submit_no_tty.py YAML "DESCRIPTION"', file=sys.stderr)
        print('   or: amlt_submit_no_tty.py YAML JOB_NAME "DESCRIPTION"',
              file=sys.stderr)
        sys.exit(2)

    patch_console()

    # Build new argv for amlt CLI
    args = ['amlt', 'run']
    if len(sys.argv) == 3:
        # YAML + DESC
        args += [sys.argv[1], '-d', sys.argv[2]]
    else:
        # YAML + JOB_NAME + DESC (and any extra args)
        args += [sys.argv[1], sys.argv[2], '-d', sys.argv[3]]
        args += sys.argv[4:]

    sys.argv = args

    from amlt.amlt import main as amlt_main
    amlt_main()


if __name__ == '__main__':
    main()
