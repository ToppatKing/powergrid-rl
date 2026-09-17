import numpy as np

class TransmissionLine:
    def __init__(self, from_bus, to_bus, reactance, max_flow_mw):
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.reactance = reactance  # AC resistance (Ohms/pu)
        self.susceptance = 1.0 / reactance # Inverse of reactance, used in DC power flow
        self.max_flow = max_flow_mw
        self.current_flow = 0.0

class GridNetwork:
    def __init__(self, num_buses=4):
        self.num_buses = num_buses
        self.lines = []
        self._build_standard_topology()

    def _build_standard_topology(self):
        """
        Creates a ring-like 4-bus topology.
        Bus 0: Coal-1, Coal-2
        Bus 1: Gas, Nuclear
        Bus 2: Wind, Solar, Battery
        Bus 3: Demand
        """
        # (from, to, reactance, max_flow_mw)
        # Low reactance = power flows more easily through this line
        self.add_line(0, 1, reactance=0.2, max_flow_mw=800)
        self.add_line(1, 2, reactance=0.25, max_flow_mw=600)
        self.add_line(2, 3, reactance=0.15, max_flow_mw=1000)
        self.add_line(3, 0, reactance=0.3, max_flow_mw=900)
        self.add_line(0, 2, reactance=0.4, max_flow_mw=500) # Cross-grid tie line

    def add_line(self, from_bus, to_bus, reactance, max_flow_mw):
        self.lines.append(TransmissionLine(from_bus, to_bus, reactance, max_flow_mw))

    def get_overloads(self):
        """Returns a list of lines exceeding their thermal limits."""
        overloads = []
        for idx, line in enumerate(self.lines):
            if abs(line.current_flow) > line.max_flow:
                overloads.append({
                    'line_idx': idx,
                    'excess_mw': abs(line.current_flow) - line.max_flow
                })
        return overloads
