import os
import subprocess

import toml

with open(
    os.path.join(os.path.dirname(__file__), "../../../pyproject.toml"), "r"
) as file:
    pyproject = toml.load(file)

# Extrae la información relevante
target_name = pyproject["project"]["name"]
version = pyproject["project"]["version"]


RELEASE_INFO = f"GMV [{target_name}] (version: {version})"


def about_msg():
    return commit_id()


def commit_id():
    file_dir = os.path.dirname(os.path.realpath(__file__))

    try:
        git_repo_path = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], cwd=file_dir
            )
            .strip()
            .decode("utf-8")
        )

        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=git_repo_path
            )
            .strip()
            .decode("utf-8")
        )

        return commit_hash
    except subprocess.CalledProcessError:
        return "N/A"
