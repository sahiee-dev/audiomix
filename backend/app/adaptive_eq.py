import numpy as np
from scipy import signal

class AdaptiveEQ:
    """
    Real-time adaptive EQ for smooth frequency transitions
    
    Research contribution: Dynamic frequency-band crossfading
    """
    
    def __init__(self, sr=44100):
        self.sr = sr
        # Define frequency bands (Hz)
        self.bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mids': (250, 500),
            'mids': (500, 2000),
            'high_mids': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
    
    def create_band_filters(self):
        """Create butterworth filters for each frequency band"""
        filters = {}
        for band_name, (low, high) in self.bands.items():
            # Bandpass filter
            sos = signal.butter(4, [low, high], 'bp', fs=self.sr, output='sos')
            filters[band_name] = sos
        return filters
    
    def analyze_frequency_balance(self, y):
        """Analyze energy distribution across frequency bands"""
        filters = self.create_band_filters()
        energy = {}
        
        for band_name, sos in filters.items():
            # Filter signal
            filtered = signal.sosfilt(sos, y)
            # Compute RMS energy
            energy[band_name] = np.sqrt(np.mean(filtered**2))
        
        return energy
    
    def create_adaptive_crossfade(self, y_a, y_b, fade_samples):
        """
        Create frequency-adaptive crossfade
        
        Strategy:
        - Low frequencies (bass): Quick swap at 40% point
        - Mid frequencies: Standard equal-power fade
        - High frequencies (hi-hats): Early fade-in of track B
        """
        min_len = min(len(y_a), len(y_b))
        y_a = y_a[:min_len]
        y_b = y_b[:min_len]
        
        filters = self.create_band_filters()
        result = np.zeros(min_len)
        
        # Process each frequency band separately
        for band_name, sos in filters.items():
            # Filter both tracks
            a_band = signal.sosfilt(sos, y_a)
            b_band = signal.sosfilt(sos, y_b)
            
            # Create band-specific fade curves
            if band_name in ['sub_bass', 'bass']:
                # Bass: Quick swap at 40%
                swap_point = int(min_len * 0.4)
                fade_out = np.ones(min_len)
                fade_out[swap_point:] = np.cos(
                    np.linspace(0, np.pi/2, min_len - swap_point)
                ) ** 4  # Very fast
                fade_in = 1 - fade_out
                
            elif band_name in ['brilliance', 'presence']:
                # Highs: Early fade (track B enters at 20%)
                start_point = int(min_len * 0.2)
                fade_in = np.zeros(min_len)
                fade_in[start_point:] = np.sin(
                    np.linspace(0, np.pi/2, min_len - start_point)
                ) ** 2
                fade_out = 1 - fade_in
                
            else:
                # Mids: Standard equal-power
                fade_out = np.cos(np.linspace(0, np.pi/2, min_len)) ** 2
                fade_in = np.sin(np.linspace(0, np.pi/2, min_len)) ** 2
            
            # Apply fades and sum to result
            result += a_band * fade_out + b_band * fade_in
        
        return result
