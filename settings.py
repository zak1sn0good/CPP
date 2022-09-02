class Setting:
  def __init__(self, latency_matrix):
    self.latency = latency_matrix
    self.capacity = [45000, 30000, 60000, 90000, 15000]
    self.ctlNum = 5
    self.swNum = 5
    self.arrivalRate = [12000 for i in range(self.swNum)]
    self.vectorX = [0, 1, 2, 3, 4]
    self.selectionVector = [1 for i in range(self.swNum)] # -1 means no controller is in that placement
    self.matrixP = [[ capacity / sum(self.capacity) for capacity in self.capacity ] for sw in range(self.swNum)]
    self.decayFactor = [0.85] * self.ctlNum
    self.beta = 1.0
    self.mu = [100.0] * 5
    self.dlt = 1.0
    self.t_thresh = 0.0706827978704 * 0.8