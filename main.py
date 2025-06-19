"""
Running example
"""
import numpy as np

from settings import DeSettings
from de_gldpc import QuantizedDE, load_lib, lib_compile

if __name__ == '__main__':
    lib_compile()
    pcm = np.loadtxt('sample_pcm.txt')
    settings = DeSettings(
            n_iterations = 20,
            input_scale=20,
            max_llr=20,
            llr_scale=0.75,
            punctured=0,
            snr=-2.0
    )
    ber = QuantizedDE(load_lib(), pcm, settings).de_min_sum()
    print(f'Evaluated bit error rate value: {ber:1.3e}')
