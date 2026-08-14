from __future__ import annotations

import click

from .train_feature import train_feature_cli


@click.group()
def main_cli():
    pass

main_cli.add_command(train_feature_cli, "train-feature")

