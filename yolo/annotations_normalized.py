# Normalized boxes: filename -> list of (label, x0, y0, x1, y1) in 0..1
# label: 'chemical_structure' or 'reaction_scheme'
ANN = {}

ANN["481016603_18077838652724108_2239786357732891810_n.png"] = [
    ("reaction_scheme", 0.14, 0.335, 0.95, 0.455),
    ("chemical_structure", 0.15, 0.345, 0.41, 0.405),
    ("chemical_structure", 0.60, 0.340, 0.86, 0.445),
    ("reaction_scheme", 0.14, 0.875, 0.99, 1.0),
    ("chemical_structure", 0.17, 0.880, 0.46, 1.0),
    ("chemical_structure", 0.61, 0.880, 0.97, 1.0),
]

ANN["ChatGPT Image Jun 19, 2026, 07_39_06 PM (1).png"] = [
    ("reaction_scheme", 0.17, 0.125, 0.93, 0.225),
    ("chemical_structure", 0.17, 0.125, 0.38, 0.205),
    ("chemical_structure", 0.39, 0.120, 0.56, 0.195),
    ("chemical_structure", 0.72, 0.125, 0.93, 0.205),
    ("chemical_structure", 0.17, 0.555, 0.48, 0.660),
    ("chemical_structure", 0.17, 0.685, 0.40, 0.800),
    ("chemical_structure", 0.43, 0.690, 0.62, 0.795),
    ("chemical_structure", 0.69, 0.685, 0.93, 0.800),
]

ANN["ChatGPT Image Jun 19, 2026, 07_39_06 PM (2).png"] = [
    ("reaction_scheme", 0.13, 0.11, 0.81, 0.25),
    ("chemical_structure", 0.13, 0.12, 0.31, 0.19),
    ("chemical_structure", 0.60, 0.11, 0.79, 0.24),
    ("chemical_structure", 0.13, 0.515, 0.31, 0.60),
    ("reaction_scheme", 0.50, 0.50, 0.97, 0.725),
    ("chemical_structure", 0.61, 0.515, 0.75, 0.585),
    ("chemical_structure", 0.79, 0.515, 0.93, 0.585),
    ("chemical_structure", 0.59, 0.625, 0.75, 0.715),
    ("chemical_structure", 0.81, 0.620, 0.96, 0.715),
    ("chemical_structure", 0.32, 0.755, 0.47, 0.825),
    ("chemical_structure", 0.19, 0.775, 0.37, 0.845),
    ("chemical_structure", 0.65, 0.755, 0.83, 0.835),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_06 PM (3).png"] = [
    ("reaction_scheme", 0.155, 0.095, 0.89, 0.21),
    ("chemical_structure", 0.155, 0.10, 0.35, 0.175),
    ("chemical_structure", 0.59, 0.10, 0.76, 0.18),
    ("chemical_structure", 0.62, 0.47, 0.81, 0.55),
    ("chemical_structure", 0.155, 0.64, 0.35, 0.725),
    ("chemical_structure", 0.39, 0.64, 0.56, 0.725),
    ("reaction_scheme", 0.57, 0.62, 0.89, 0.795),
    ("chemical_structure", 0.59, 0.635, 0.79, 0.725),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (4).png"] = [
    ("reaction_scheme", 0.15, 0.185, 0.95, 0.325),
    ("chemical_structure", 0.155, 0.19, 0.33, 0.285),
    ("chemical_structure", 0.49, 0.19, 0.66, 0.295),
    ("chemical_structure", 0.67, 0.19, 0.84, 0.305),
    ("reaction_scheme", 0.61, 0.345, 0.94, 0.475),
    ("chemical_structure", 0.63, 0.35, 0.79, 0.455),
    ("chemical_structure", 0.79, 0.35, 0.94, 0.455),
    ("chemical_structure", 0.40, 0.715, 0.56, 0.805),
    ("chemical_structure", 0.55, 0.715, 0.71, 0.815),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (5).png"] = [
    ("reaction_scheme", 0.12, 0.125, 0.95, 0.225),
    ("chemical_structure", 0.12, 0.125, 0.31, 0.205),
    ("chemical_structure", 0.52, 0.125, 0.67, 0.205),
    ("chemical_structure", 0.69, 0.150, 0.84, 0.185),
    ("reaction_scheme", 0.12, 0.560, 0.89, 0.665),
    ("chemical_structure", 0.12, 0.570, 0.28, 0.655),
    ("chemical_structure", 0.37, 0.570, 0.53, 0.655),
    ("chemical_structure", 0.65, 0.570, 0.84, 0.655),
    ("reaction_scheme", 0.24, 0.675, 0.89, 0.765),
    ("chemical_structure", 0.53, 0.675, 0.69, 0.760),
    ("chemical_structure", 0.77, 0.685, 0.91, 0.725),
    ("chemical_structure", 0.13, 0.785, 0.37, 0.925),
    ("chemical_structure", 0.81, 0.785, 0.96, 0.865),
    ("chemical_structure", 0.81, 0.875, 0.97, 0.955),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (6).png"] = [
    ("reaction_scheme", 0.15, 0.095, 0.79, 0.23),
    ("chemical_structure", 0.15, 0.095, 0.31, 0.205),
    ("chemical_structure", 0.57, 0.095, 0.73, 0.215),
    ("reaction_scheme", 0.12, 0.505, 0.93, 0.605),
    ("chemical_structure", 0.12, 0.505, 0.28, 0.605),
    ("chemical_structure", 0.39, 0.510, 0.56, 0.600),
    ("chemical_structure", 0.61, 0.510, 0.77, 0.600),
    ("chemical_structure", 0.51, 0.620, 0.69, 0.740),
    ("chemical_structure", 0.13, 0.825, 0.25, 0.915),
    ("chemical_structure", 0.27, 0.820, 0.43, 0.915),
    ("chemical_structure", 0.43, 0.820, 0.59, 0.920),
    ("chemical_structure", 0.61, 0.820, 0.76, 0.915),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (7).png"] = [
    ("reaction_scheme", 0.16, 0.125, 0.86, 0.235),
    ("chemical_structure", 0.17, 0.135, 0.31, 0.215),
    ("chemical_structure", 0.35, 0.125, 0.51, 0.230),
    ("chemical_structure", 0.65, 0.130, 0.83, 0.225),
    ("chemical_structure", 0.17, 0.300, 0.35, 0.395),
    ("chemical_structure", 0.45, 0.300, 0.63, 0.395),
    ("reaction_scheme", 0.10, 0.685, 0.93, 0.795),
    ("chemical_structure", 0.11, 0.690, 0.37, 0.795),
    ("chemical_structure", 0.54, 0.690, 0.71, 0.785),
    ("chemical_structure", 0.77, 0.690, 0.92, 0.775),
    ("chemical_structure", 0.12, 0.845, 0.28, 0.955),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (8).png"] = [
    ("reaction_scheme", 0.11, 0.125, 0.87, 0.245),
    ("chemical_structure", 0.12, 0.140, 0.28, 0.190),
    ("chemical_structure", 0.32, 0.125, 0.51, 0.210),
    ("chemical_structure", 0.65, 0.130, 0.83, 0.190),
    ("reaction_scheme", 0.30, 0.620, 0.93, 0.835),
    ("chemical_structure", 0.31, 0.635, 0.47, 0.785),
    ("chemical_structure", 0.49, 0.635, 0.67, 0.785),
    ("chemical_structure", 0.67, 0.645, 0.93, 0.735),
    ("reaction_scheme", 0.41, 0.845, 0.93, 0.975),
    ("chemical_structure", 0.47, 0.850, 0.61, 0.930),
    ("chemical_structure", 0.72, 0.850, 0.89, 0.925),
    ("chemical_structure", 0.11, 0.905, 0.31, 0.985),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_07 PM (9).png"] = [
    ("reaction_scheme", 0.13, 0.115, 0.79, 0.245),
    ("chemical_structure", 0.13, 0.115, 0.29, 0.210),
    ("chemical_structure", 0.24, 0.150, 0.43, 0.195),
    ("chemical_structure", 0.57, 0.115, 0.75, 0.240),
    ("reaction_scheme", 0.33, 0.655, 0.93, 0.785),
    ("chemical_structure", 0.34, 0.660, 0.48, 0.750),
    ("chemical_structure", 0.54, 0.665, 0.67, 0.765),
    ("chemical_structure", 0.70, 0.660, 0.86, 0.765),
    ("chemical_structure", 0.12, 0.795, 0.29, 0.885),
    ("chemical_structure", 0.36, 0.785, 0.53, 0.900),
]
ANN["ChatGPT Image Jun 19, 2026, 07_39_08 PM (10).png"] = [
    ("reaction_scheme", 0.15, 0.175, 0.91, 0.300),
    ("chemical_structure", 0.15, 0.180, 0.31, 0.285),
    ("chemical_structure", 0.29, 0.180, 0.45, 0.285),
    ("chemical_structure", 0.57, 0.195, 0.86, 0.285),
    ("reaction_scheme", 0.12, 0.355, 0.93, 0.455),
    ("chemical_structure", 0.13, 0.355, 0.28, 0.455),
    ("chemical_structure", 0.34, 0.355, 0.53, 0.455),
    ("chemical_structure", 0.64, 0.355, 0.92, 0.455),
    ("reaction_scheme", 0.12, 0.470, 0.79, 0.555),
    ("chemical_structure", 0.19, 0.470, 0.43, 0.555),
    ("chemical_structure", 0.59, 0.475, 0.79, 0.525),
    ("chemical_structure", 0.65, 0.570, 0.83, 0.650),
    ("chemical_structure", 0.65, 0.655, 0.83, 0.735),
    ("chemical_structure", 0.61, 0.755, 0.91, 0.825),
]

ANN["Screenshot 2026-06-19 at 7.44.32 PM.png"] = [
    ("reaction_scheme", 0.12, 0.125, 0.80, 0.275),
    ("chemical_structure", 0.12, 0.14, 0.31, 0.265),
    ("chemical_structure", 0.33, 0.155, 0.53, 0.270),
    ("chemical_structure", 0.16, 0.70, 0.44, 0.825),
]
ANN["Screenshot 2026-06-19 at 7.47.52 PM.png"] = [
    ("reaction_scheme", 0.12, 0.115, 0.81, 0.225),
    ("chemical_structure", 0.12, 0.125, 0.31, 0.215),
    ("chemical_structure", 0.31, 0.130, 0.47, 0.200),
    ("chemical_structure", 0.64, 0.125, 0.81, 0.205),
    ("chemical_structure", 0.12, 0.685, 0.42, 0.805),
]
ANN["Screenshot 2026-06-19 at 7.48.38 PM.png"] = [
    ("reaction_scheme", 0.13, 0.175, 0.90, 0.305),
    ("chemical_structure", 0.13, 0.18, 0.42, 0.295),
    ("chemical_structure", 0.70, 0.18, 0.91, 0.295),
    ("chemical_structure", 0.14, 0.655, 0.38, 0.785),
]
ANN["Screenshot 2026-06-19 at 7.48.47 PM.png"] = [
    ("reaction_scheme", 0.13, 0.155, 0.88, 0.275),
    ("chemical_structure", 0.13, 0.16, 0.38, 0.275),
    ("chemical_structure", 0.68, 0.16, 0.90, 0.275),
    ("chemical_structure", 0.14, 0.705, 0.46, 0.825),
]
ANN["Screenshot 2026-06-19 at 7.49.04 PM.png"] = [
    ("reaction_scheme", 0.09, 0.155, 0.93, 0.245),
    ("chemical_structure", 0.09, 0.155, 0.43, 0.235),
    ("chemical_structure", 0.61, 0.155, 0.93, 0.235),
    ("reaction_scheme", 0.15, 0.265, 0.93, 0.350),
    ("chemical_structure", 0.55, 0.265, 0.93, 0.350),
    ("chemical_structure", 0.12, 0.705, 0.56, 0.815),
]
ANN["Screenshot 2026-06-19 at 7.49.13 PM.png"] = [
    ("reaction_scheme", 0.13, 0.14, 0.83, 0.27),
    ("chemical_structure", 0.13, 0.145, 0.33, 0.255),
    ("chemical_structure", 0.62, 0.14, 0.83, 0.265),
    ("chemical_structure", 0.15, 0.685, 0.40, 0.815),
]
ANN["Screenshot 2026-06-19 at 7.49.22 PM.png"] = [
    ("reaction_scheme", 0.09, 0.135, 0.93, 0.265),
    ("chemical_structure", 0.09, 0.155, 0.27, 0.235),
    ("chemical_structure", 0.31, 0.135, 0.49, 0.260),
    ("chemical_structure", 0.68, 0.135, 0.93, 0.260),
    ("chemical_structure", 0.13, 0.695, 0.47, 0.835),
]
ANN["Screenshot 2026-06-19 at 7.49.29 PM.png"] = [
    ("reaction_scheme", 0.09, 0.135, 0.93, 0.225),
    ("chemical_structure", 0.09, 0.150, 0.35, 0.215),
    ("chemical_structure", 0.68, 0.150, 0.93, 0.220),
    ("reaction_scheme", 0.08, 0.355, 0.86, 0.485),
    ("chemical_structure", 0.08, 0.360, 0.36, 0.475),
    ("chemical_structure", 0.49, 0.360, 0.73, 0.475),
    ("reaction_scheme", 0.08, 0.515, 0.86, 0.605),
    ("chemical_structure", 0.49, 0.515, 0.81, 0.600),
    ("chemical_structure", 0.12, 0.795, 0.51, 0.875),
]
ANN["Screenshot 2026-06-19 at 7.49.36 PM.png"] = [
    ("reaction_scheme", 0.12, 0.135, 0.86, 0.255),
    ("chemical_structure", 0.12, 0.145, 0.32, 0.255),
    ("chemical_structure", 0.33, 0.150, 0.53, 0.235),
    ("chemical_structure", 0.68, 0.135, 0.89, 0.255),
    ("chemical_structure", 0.18, 0.685, 0.48, 0.805),
]
ANN["Screenshot 2026-06-19 at 7.49.43 PM.png"] = [
    ("reaction_scheme", 0.08, 0.135, 0.93, 0.345),
    ("chemical_structure", 0.09, 0.145, 0.49, 0.220),
    ("chemical_structure", 0.09, 0.270, 0.43, 0.340),
    ("reaction_scheme", 0.07, 0.435, 0.96, 0.565),
    ("chemical_structure", 0.09, 0.445, 0.31, 0.555),
    ("chemical_structure", 0.32, 0.445, 0.53, 0.555),
    ("chemical_structure", 0.54, 0.445, 0.73, 0.555),
    ("chemical_structure", 0.77, 0.440, 0.96, 0.555),
    ("reaction_scheme", 0.17, 0.605, 0.81, 0.715),
    ("chemical_structure", 0.19, 0.610, 0.41, 0.710),
    ("chemical_structure", 0.54, 0.605, 0.79, 0.710),
    ("chemical_structure", 0.12, 0.835, 0.53, 0.920),
]

ANN["abc.png"] = [
    ("reaction_scheme", 0.26, 0.265, 0.74, 0.375),
    ("chemical_structure", 0.26, 0.265, 0.41, 0.365),
    ("chemical_structure", 0.54, 0.265, 0.73, 0.375),
    ("reaction_scheme", 0.06, 0.595, 0.93, 0.755),
    ("chemical_structure", 0.06, 0.600, 0.23, 0.745),
    ("chemical_structure", 0.26, 0.600, 0.43, 0.745),
    ("chemical_structure", 0.44, 0.600, 0.61, 0.745),
    ("chemical_structure", 0.62, 0.600, 0.76, 0.745),
    ("chemical_structure", 0.78, 0.600, 0.93, 0.700),
    ("chemical_structure", 0.61, 0.820, 0.81, 0.935),
]
ANN["dc71d4c197836c9ba61b12b24a6ec4b8.png"] = [
    ("reaction_scheme", 0.02, 0.035, 0.93, 0.165),
    ("chemical_structure", 0.02, 0.065, 0.21, 0.150),
    ("chemical_structure", 0.21, 0.055, 0.40, 0.140),
    ("chemical_structure", 0.58, 0.050, 0.92, 0.150),
    ("reaction_scheme", 0.02, 0.195, 0.98, 0.420),
    ("chemical_structure", 0.02, 0.205, 0.19, 0.315),
    ("chemical_structure", 0.34, 0.205, 0.53, 0.310),
    ("chemical_structure", 0.54, 0.205, 0.73, 0.315),
    ("chemical_structure", 0.77, 0.205, 0.98, 0.315),
    ("chemical_structure", 0.61, 0.355, 0.89, 0.455),
    ("chemical_structure", 0.02, 0.555, 0.19, 0.655),
    ("chemical_structure", 0.21, 0.555, 0.39, 0.655),
    ("chemical_structure", 0.41, 0.555, 0.59, 0.655),
    ("chemical_structure", 0.60, 0.555, 0.75, 0.655),
    ("chemical_structure", 0.76, 0.550, 0.96, 0.655),
    ("reaction_scheme", 0.02, 0.685, 0.63, 0.805),
    ("chemical_structure", 0.04, 0.685, 0.23, 0.795),
    ("chemical_structure", 0.29, 0.685, 0.45, 0.775),
    ("chemical_structure", 0.49, 0.685, 0.67, 0.805),
    ("reaction_scheme", 0.02, 0.815, 0.63, 0.935),
    ("chemical_structure", 0.02, 0.815, 0.21, 0.935),
    ("chemical_structure", 0.27, 0.815, 0.47, 0.905),
    ("chemical_structure", 0.49, 0.815, 0.69, 0.935),
]
ANN["dfvr.png"] = [
    ("reaction_scheme", 0.11, 0.155, 0.89, 0.245),
    ("chemical_structure", 0.11, 0.155, 0.25, 0.245),
    ("chemical_structure", 0.57, 0.155, 0.73, 0.245),
    ("reaction_scheme", 0.12, 0.555, 0.83, 0.685),
    ("chemical_structure", 0.13, 0.560, 0.39, 0.675),
    ("chemical_structure", 0.54, 0.570, 0.81, 0.690),
]
ANN["f8z2b6smafq11.png"] = [
    ("reaction_scheme", 0.18, 0.175, 0.56, 0.245),
    ("chemical_structure", 0.19, 0.175, 0.31, 0.225),
    ("chemical_structure", 0.41, 0.185, 0.56, 0.245),
    ("reaction_scheme", 0.17, 0.275, 0.97, 0.360),
    ("chemical_structure", 0.17, 0.285, 0.34, 0.345),
    ("chemical_structure", 0.57, 0.285, 0.71, 0.345),
    ("chemical_structure", 0.84, 0.270, 0.98, 0.350),
    ("reaction_scheme", 0.17, 0.415, 0.98, 0.485),
    ("chemical_structure", 0.17, 0.415, 0.31, 0.475),
    ("chemical_structure", 0.61, 0.415, 0.79, 0.485),
    ("chemical_structure", 0.87, 0.415, 0.99, 0.485),
    ("reaction_scheme", 0.12, 0.545, 0.93, 0.655),
    ("chemical_structure", 0.13, 0.555, 0.31, 0.620),
    ("chemical_structure", 0.37, 0.550, 0.56, 0.655),
    ("chemical_structure", 0.59, 0.545, 0.76, 0.625),
    ("chemical_structure", 0.81, 0.545, 0.96, 0.620),
]
ANN["il_794xN.5175845357_lxtv.png"] = [
    ("reaction_scheme", 0.01, 0.095, 0.46, 0.220),
    ("chemical_structure", 0.01, 0.115, 0.14, 0.215),
    ("chemical_structure", 0.17, 0.095, 0.31, 0.215),
    ("chemical_structure", 0.35, 0.095, 0.49, 0.215),
    ("reaction_scheme", 0.01, 0.355, 0.82, 0.550),
    ("chemical_structure", 0.01, 0.375, 0.17, 0.535),
    ("chemical_structure", 0.23, 0.375, 0.41, 0.525),
    ("chemical_structure", 0.47, 0.375, 0.63, 0.525),
    ("chemical_structure", 0.67, 0.375, 0.83, 0.535),
    ("reaction_scheme", 0.01, 0.615, 0.42, 0.755),
    ("chemical_structure", 0.01, 0.635, 0.15, 0.745),
    ("chemical_structure", 0.21, 0.615, 0.35, 0.755),
]
_amine = [
    ("reaction_scheme", 0.09, 0.230, 0.92, 0.295),
    ("reaction_scheme", 0.09, 0.300, 0.63, 0.355),
    ("reaction_scheme", 0.09, 0.365, 0.80, 0.435),
    ("reaction_scheme", 0.09, 0.450, 0.93, 0.535),
    ("reaction_scheme", 0.10, 0.605, 0.61, 0.720),
    ("chemical_structure", 0.11, 0.610, 0.27, 0.715),
    ("chemical_structure", 0.40, 0.610, 0.57, 0.715),
    ("reaction_scheme", 0.07, 0.755, 0.93, 0.825),
    ("reaction_scheme", 0.07, 0.845, 0.93, 0.925),
]
ANN["jytdfj.png"] = list(_amine)
ANN["kytfvtgt.png"] = list(_amine)
ANN["page-1.png"] = [
    ("reaction_scheme", 0.11, 0.175, 0.86, 0.305),
    ("chemical_structure", 0.11, 0.180, 0.29, 0.285),
    ("chemical_structure", 0.29, 0.175, 0.47, 0.275),
    ("chemical_structure", 0.64, 0.180, 0.86, 0.305),
    ("chemical_structure", 0.15, 0.745, 0.51, 0.880),
]
ANN["page-10.png"] = [
    ("reaction_scheme", 0.12, 0.155, 0.86, 0.245),
    ("chemical_structure", 0.12, 0.155, 0.31, 0.245),
    ("chemical_structure", 0.66, 0.155, 0.86, 0.245),
    ("reaction_scheme", 0.14, 0.265, 0.63, 0.405),
    ("chemical_structure", 0.38, 0.275, 0.63, 0.405),
    ("chemical_structure", 0.15, 0.735, 0.45, 0.885),
]
ANN["page-2.png"] = [
    ("reaction_scheme", 0.11, 0.155, 0.86, 0.255),
    ("chemical_structure", 0.11, 0.155, 0.31, 0.255),
    ("chemical_structure", 0.62, 0.155, 0.86, 0.255),
    ("reaction_scheme", 0.07, 0.395, 0.91, 0.555),
    ("chemical_structure", 0.07, 0.400, 0.37, 0.525),
    ("chemical_structure", 0.60, 0.395, 0.91, 0.555),
]
ANN["page-3.png"] = [
    ("reaction_scheme", 0.12, 0.155, 0.91, 0.245),
    ("chemical_structure", 0.12, 0.160, 0.27, 0.240),
    ("chemical_structure", 0.28, 0.160, 0.43, 0.240),
    ("reaction_scheme", 0.12, 0.295, 0.91, 0.405),
    ("chemical_structure", 0.12, 0.300, 0.35, 0.400),
    ("chemical_structure", 0.61, 0.295, 0.87, 0.395),
    ("reaction_scheme", 0.09, 0.495, 0.86, 0.600),
    ("chemical_structure", 0.11, 0.500, 0.31, 0.595),
    ("chemical_structure", 0.54, 0.500, 0.80, 0.595),
    ("reaction_scheme", 0.04, 0.615, 0.86, 0.720),
    ("chemical_structure", 0.09, 0.620, 0.33, 0.715),
    ("chemical_structure", 0.54, 0.620, 0.80, 0.715),
]
ANN["page-4.png"] = [
    ("reaction_scheme", 0.09, 0.175, 0.91, 0.275),
    ("chemical_structure", 0.09, 0.180, 0.31, 0.270),
    ("chemical_structure", 0.60, 0.180, 0.83, 0.270),
    ("chemical_structure", 0.17, 0.675, 0.56, 0.790),
]
ANN["page-5.png"] = [
    ("reaction_scheme", 0.09, 0.155, 0.92, 0.285),
    ("chemical_structure", 0.09, 0.155, 0.27, 0.265),
    ("chemical_structure", 0.53, 0.155, 0.72, 0.285),
    ("chemical_structure", 0.73, 0.155, 0.91, 0.285),
    ("chemical_structure", 0.29, 0.415, 0.51, 0.555),
]
ANN["page-6.png"] = [
    ("reaction_scheme", 0.12, 0.155, 0.86, 0.275),
    ("chemical_structure", 0.12, 0.160, 0.32, 0.270),
    ("chemical_structure", 0.64, 0.155, 0.86, 0.265),
    ("chemical_structure", 0.15, 0.685, 0.43, 0.820),
]
ANN["page-7.png"] = [
    ("reaction_scheme", 0.09, 0.155, 0.93, 0.245),
    ("chemical_structure", 0.09, 0.155, 0.43, 0.235),
    ("chemical_structure", 0.61, 0.155, 0.93, 0.235),
    ("reaction_scheme", 0.11, 0.295, 0.93, 0.385),
    ("chemical_structure", 0.54, 0.295, 0.93, 0.385),
]
ANN["page-8.png"] = [
    ("reaction_scheme", 0.12, 0.155, 0.86, 0.275),
    ("chemical_structure", 0.12, 0.160, 0.33, 0.270),
    ("chemical_structure", 0.64, 0.160, 0.86, 0.270),
    ("chemical_structure", 0.15, 0.715, 0.41, 0.850),
]
ANN["page-9.png"] = [
    ("reaction_scheme", 0.09, 0.155, 0.91, 0.275),
    ("chemical_structure", 0.09, 0.160, 0.28, 0.255),
    ("chemical_structure", 0.31, 0.155, 0.51, 0.275),
    ("chemical_structure", 0.68, 0.155, 0.91, 0.275),
    ("chemical_structure", 0.13, 0.775, 0.51, 0.945),
]

ANN["scfdsfver.png"] = [
    ("reaction_scheme", 0.09, 0.065, 0.92, 0.200),
    ("chemical_structure", 0.12, 0.070, 0.31, 0.180),
    ("chemical_structure", 0.39, 0.075, 0.56, 0.165),
    ("chemical_structure", 0.64, 0.070, 0.92, 0.175),
    ("reaction_scheme", 0.12, 0.395, 0.92, 0.525),
    ("chemical_structure", 0.13, 0.400, 0.33, 0.500),
    ("chemical_structure", 0.64, 0.400, 0.89, 0.525),
    ("reaction_scheme", 0.09, 0.625, 0.91, 0.760),
    ("chemical_structure", 0.09, 0.630, 0.35, 0.755),
    ("chemical_structure", 0.65, 0.630, 0.91, 0.760),
]
ANN["some-fragments-of-my-notes-from-organic-chemistry-in-v0-7tdomeyhr6581.png"] = [
    ("reaction_scheme", 0.15, 0.115, 0.86, 0.185),
    ("chemical_structure", 0.16, 0.115, 0.27, 0.180),
    ("chemical_structure", 0.29, 0.115, 0.51, 0.180),
    ("chemical_structure", 0.55, 0.115, 0.76, 0.185),
    ("reaction_scheme", 0.04, 0.345, 0.89, 0.465),
    ("chemical_structure", 0.05, 0.350, 0.18, 0.460),
    ("chemical_structure", 0.20, 0.350, 0.40, 0.450),
    ("chemical_structure", 0.50, 0.350, 0.66, 0.450),
    ("chemical_structure", 0.70, 0.350, 0.88, 0.460),
    ("reaction_scheme", 0.04, 0.480, 0.89, 0.600),
    ("chemical_structure", 0.05, 0.485, 0.22, 0.595),
    ("chemical_structure", 0.26, 0.485, 0.46, 0.585),
    ("chemical_structure", 0.50, 0.485, 0.66, 0.585),
    ("chemical_structure", 0.70, 0.485, 0.88, 0.595),
    ("reaction_scheme", 0.04, 0.915, 0.91, 0.975),
]
ANN["some-of-my-notes-from-drug-synthesis-2018-2019-v0-hgwiwoh3lve81.png"] = [
    ("reaction_scheme", 0.18, 0.105, 0.72, 0.215),
    ("reaction_scheme", 0.18, 0.265, 0.78, 0.355),
    ("chemical_structure", 0.18, 0.265, 0.33, 0.350),
    ("chemical_structure", 0.45, 0.265, 0.61, 0.350),
    ("reaction_scheme", 0.18, 0.385, 0.70, 0.500),
    ("reaction_scheme", 0.10, 0.560, 0.92, 0.660),
    ("chemical_structure", 0.11, 0.565, 0.27, 0.655),
    ("chemical_structure", 0.29, 0.565, 0.45, 0.655),
    ("chemical_structure", 0.49, 0.565, 0.65, 0.655),
    ("chemical_structure", 0.67, 0.560, 0.84, 0.655),
    ("reaction_scheme", 0.18, 0.710, 0.82, 0.790),
    ("reaction_scheme", 0.18, 0.860, 0.72, 0.930),
]

ANN["ty6frt5fkuy.png"] = [
    ("reaction_scheme", 0.09, 0.035, 0.86, 0.105),
    ("reaction_scheme", 0.10, 0.215, 0.93, 0.285),
    ("reaction_scheme", 0.09, 0.335, 0.72, 0.450),
    ("chemical_structure", 0.09, 0.345, 0.25, 0.445),
    ("chemical_structure", 0.39, 0.345, 0.56, 0.445),
    ("reaction_scheme", 0.09, 0.505, 0.80, 0.625),
    ("reaction_scheme", 0.09, 0.665, 0.80, 0.765),
    ("reaction_scheme", 0.09, 0.815, 0.91, 0.905),
]
ANN["yuytiy.png"] = [
    ("reaction_scheme", 0.04, 0.125, 0.70, 0.175),
    ("reaction_scheme", 0.04, 0.195, 0.56, 0.270),
    ("chemical_structure", 0.04, 0.195, 0.19, 0.270),
    ("chemical_structure", 0.39, 0.195, 0.55, 0.270),
    ("reaction_scheme", 0.04, 0.280, 0.56, 0.350),
    ("chemical_structure", 0.04, 0.280, 0.20, 0.350),
    ("chemical_structure", 0.39, 0.280, 0.56, 0.350),
    ("reaction_scheme", 0.04, 0.360, 0.56, 0.430),
    ("chemical_structure", 0.04, 0.360, 0.20, 0.430),
    ("chemical_structure", 0.39, 0.360, 0.56, 0.430),
    ("reaction_scheme", 0.04, 0.440, 0.56, 0.520),
    ("chemical_structure", 0.04, 0.440, 0.21, 0.520),
    ("chemical_structure", 0.38, 0.440, 0.56, 0.520),
    ("reaction_scheme", 0.04, 0.545, 0.76, 0.605),
    ("reaction_scheme", 0.04, 0.625, 0.71, 0.690),
    ("reaction_scheme", 0.04, 0.735, 0.86, 0.855),
    ("chemical_structure", 0.04, 0.755, 0.19, 0.845),
    ("chemical_structure", 0.43, 0.740, 0.61, 0.825),
    ("chemical_structure", 0.71, 0.770, 0.89, 0.880),
    ("reaction_scheme", 0.04, 0.895, 0.86, 0.985),
    ("chemical_structure", 0.04, 0.905, 0.21, 0.985),
    ("chemical_structure", 0.43, 0.895, 0.62, 0.985),
]
