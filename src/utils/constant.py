
EXPECTED_COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

SENSOR_COLUMNS=[
    *[f"sensor_{i}" for i in range(1, 22)],
]