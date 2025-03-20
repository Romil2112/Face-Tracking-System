"""
Executable Build Script for Face Tracking Application
Author: Romil V. Shah
This script builds an executable version of the face tracking application using PyInstaller.
"""

from PyInstaller.__main__ import run

pyinstaller_args = [
    '--onefile',
    '--name=FaceTrackingApp',
    '--distpath=.',
    '--workpath=build',
    '--specpath=build',
    'src/main.py'
]

run(pyinstaller_args)
