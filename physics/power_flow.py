import numpy as np

class DCPowerFlow:
    def __init__(self, network):
        self.network = network
        self.B_matrix = self._build_b_matrix()
        
        # To solve DC power flow, we must remove the slack bus (Bus 0) 
        # to prevent a singular matrix (infinite solutions).
        self.B_matrix_reduced = self.B_matrix[1:, 1:]
        self.B_inv = np.linalg.inv(self.B_matrix_reduced)

    def _build_b_matrix(self):
        """Builds the susceptance matrix (B-matrix) for the grid."""
        n = self.network.num_buses
        B = np.zeros((n, n))
        
        for line in self.network.lines:
            i, j = line.from_bus, line.to_bus
            b = line.susceptance
            
            B[i, j] -= b
            B[j, i] -= b
            B[i, i] += b
            B[j, j] += b
            
        return B

    def solve(self, net_injections):
        """
        Calculates power flow through all lines based on nodal injections.
        net_injections: array of length num_buses. (Generation - Load) at each bus.
        Must sum to ~0 (Total Generation = Total Load).
        """
        # P = B * theta -> theta = B_inv * P
        # Remove slack bus injection (Bus 0) for the solve
        P_reduced = net_injections[1:]
        
        # Calculate voltage angles (theta) at each bus
        theta_reduced = self.B_inv @ P_reduced
        theta = np.zeros(self.network.num_buses)
        theta[1:] = theta_reduced # Slack bus angle is 0
        
        # Calculate actual flow on each line: Flow_ij = B_ij * (theta_i - theta_j)
        for line in self.network.lines:
            i, j = line.from_bus, line.to_bus
            line.current_flow = line.susceptance * (theta[i] - theta[j])
            
        return theta
