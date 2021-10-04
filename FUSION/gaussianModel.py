class GaussianModel:
    def __init__(self, alphah=None, refPoints=None):
        self.alphah = alphah
        self.refPoints = refPoints

    def setAlpha(self, alphah):
        self.alphah = alphah

    def setRefPoints(self, refPoints):
        self.refPoints = refPoints