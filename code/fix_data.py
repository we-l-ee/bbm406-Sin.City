import subprocess as sp
''''
FFMPEG_BIN = "ffmpeg.exe"

command = [ FFMPEG_BIN,
        '-i', "F:\\d_0.ogg",
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-ar', '22050', # ouput will have 44100 Hz
        '-ac', '1', # stereo (set to '1' for mono)
        '-']
pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

raw_audio = pipe.stdout.read(88200*4)
audio_array = np.fromstring(raw_audio, dtype="float32")
# audio_array = audio_array.reshape((int(len(audio_array)/2), 2))
print(audio_array)

import numpy as np
import audioread as ar
import scipy.signal
y = []
with ar.audio_open( "F:\\d_0.ogg") as file:
    sr_native = file.samplerate
    n_channels = file.channels

    for f in file:
        scale = 1. / float(1 << ((8 * 2) - 1))

        # Construct the format string
        fmt = '<i{:d}'.format(2)
        a = scale * np.frombuffer(f, fmt).astype(np.float32)
        y.append(a)

    y = np.concatenate(y)
    print(sr_native)
    if n_channels > 1:
        y = y.reshape((-1, n_channels)).T
        y = np.mean(y, axis=0)
        if sr_native == 22050:
            pass
        else:
            y_hat = scipy.signal.resample(y, 22050, axis=-1)
            n = y_hat.shape[-1]
            if n > 22050:
                slices = [slice(None)] * y_hat.ndim
                slices[-1] = slice(0, 22050)
                y_hat = y_hat[slices]

            elif n < 22050:
                lengths = [(0, 0)] * y_hat.ndim
                lengths[-1] = (0, 22050 - n)
                y_hat = np.pad(y_hat, lengths)
            y = np.ascontiguousarray(y_hat, dtype=y.dtype)

        y = np.ascontiguousarray(y, dtype=np.float32)
    print(y.shape)
'''



