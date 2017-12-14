import os
import librosa as lb
import argparse
import tensorflow as tf

os.chdir('..')
path = "datasets\\ege\\e_0.ogg"

y,sr = lb.core.load(path)
