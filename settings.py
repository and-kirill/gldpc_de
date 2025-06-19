from dataclasses import dataclass


@dataclass
class DeSettings:
    """
    Parameters of Quantized density evolution with hard input
    """
    # The number of density evolution iterations
    n_iterations: int
    # How many LLR values enclosed in [0, 1] interval
    input_scale: int
    # Maximum LLR value. All LLRs that exceed this value will be clipped
    # Quantized DE contains of max_llr * input_scale points
    max_llr: int
    # LLR scale of the normalized min-sum
    llr_scale: float
    # The number of punctured positions. Puncturing is applied to first columns
    punctured: int
    # Signal to Noise Ratio (dB)
    snr: float

    def __str__(self):
        n_points = self.max_llr * self.input_scale * 2 + 1
        msg = ''
        msg += f'  The number of iterations:        {self.n_iterations}\n'
        msg += f'  Input LLR scale:                 {self.input_scale}\n'
        msg += f'  Maximum LLR value:               {self.max_llr} (PDF of {n_points} points)\n'
        msg += f'  Normalized min-sum scale:        {self.llr_scale:1.3f}\n'
        msg += f'  The number of punctured columns: {self.punctured}\n'
        msg += f' Signal to noise ratio:            {self.snr:1.3f} dB\n'
        return msg
