import numpy as np

class Point:


    def __init__(self, type, x = None, y = None, z = None):
        
        self.TYPE = "NOT DEFINED" if type is None else type

        if x is None:
            # call the function to set random coordinate value
            self.set_ran()
        else:
            # If arguments are provided, use them (default is 0)
            self.X = x
            self.Y = y if y is not None else 0
            self.Z = z if z is not None else 0

    def set_ran(self):
        SCALE = 1
        # Generates 3 random numbers between 0 and 1
        self.X, self.Y, self.Z = np.random.rand(3) * SCALE

    def info(self):
        print(f"{self.TYPE} : ({self.X:.2f}, {self.Y:.2f}, {self.Z:.2f})")

    def euc_dist(self, other):
        # 'other' should be another Point object
        return ((other.X - self.X)**2 + 
                (other.Y - self.Y)**2 + 
                (other.Z - self.Z)**2) ** 0.5
    

