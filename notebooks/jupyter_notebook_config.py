# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
# https://towardsdatascience.com/version-control-for-jupyter-notebook-3e6cef13392d

import os
from subprocess import check_call


def post_save(model, os_path, contents_manager):
    """Jupyter post-save hook for converting notebooks to .py scripts on save."""

    if model['type'] != 'notebook':
        return  # only do this for notebooks

    working_directory, filename = os.path.split(os_path)

    # Launch a thread to save the notebook as a Python script in the right folder
    check_call(['jupyter', 'nbconvert', '--output-dir={}'.format("../scripts/"), '--to',
                'script', filename], cwd=working_directory)

    # Launch a thread to save the notebook as a HTML page in the right folder
    # check_call(['jupyter', 'nbconvert', '--output-dir={}'.format("../scripts/"), '--to',
    #             'html', filename], cwd=working_directory)


c.FileContentsManager.post_save_hook = post_save
