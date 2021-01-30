# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
# https://towardsdatascience.com/version-control-for-jupyter-notebook-3e6cef13392d

import os
from subprocess import check_call


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return  # only do this for notebooks

    directory, filename = os.path.split(os_path)

    # Save the notebook as a Python script in the right folder
    check_call(['jupyter', 'nbconvert', '--to',
                'script', filename], cwd=directory)

    # Save the notebook as a HTML page in the right folder
    # check_call(['jupyter', 'nbconvert', '--to',
    #             'html', filename], cwd=directory)


c.FileContentsManager.post_save_hook = post_save
