"""
Create a TimeSeriesAnalyzer class that stores timestamp-value pairs and provides methods to calculate moving average and detect trend direction (increasing/decreasing/stable).
"""

class TimeSeriesAnalyzer:
    def __init__(self, data):
        """Initialize with list of (timestamp, value) tuples"""
        self.data = data
        self.timestamps = [item[0] for item in data]
        self.values = [item[1] for item in data]
    
    def moving_average(self, window=3):
        """Calculate moving average with specified window size"""
        moving_avgs = []
        for i in range(len(self.values) - window + 1):
            window_values = self.values[i:i + window]
            avg = sum(window_values) / window
            moving_avgs.append(round(avg, 2))
        return moving_avgs
    
    def detect_trend(self):
        """Detect overall trend direction"""
        first_half_avg = sum(self.values[:len(self.values)//2]) / (len(self.values)//2)
        second_half_avg = sum(self.values[len(self.values)//2:]) / (len(self.values) - len(self.values)//2)
        
        threshold = 0.02  # 2% threshold for stability
        diff_percentage = (second_half_avg - first_half_avg) / first_half_avg
        
        if diff_percentage > threshold:
            return 'increasing'
        elif diff_percentage < -threshold:
            return 'decreasing'
        else:
            return 'stable'

# Test
data = [(1, 100), (2, 105), (3, 110), (4, 108), (5, 115)]
analyzer = TimeSeriesAnalyzer(data)
print("Moving Average:", analyzer.moving_average(window=3))
print("Trend:", analyzer.detect_trend())
# Output: Moving Average: [105.0, 107.67, 111.0]
# Output: Trend: 'increasing'

