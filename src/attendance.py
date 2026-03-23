import csv
import os
from datetime import datetime

ATTENDANCE_FILE = os.path.join("output", "attendance.csv")
_logged_names = None


def _load_logged_names():
    global _logged_names

    if _logged_names is None:
        _logged_names = set()

        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        _logged_names.add(row[0].strip())

    return _logged_names


def mark_attendance(name):
    if not name or name == "Unknown":
        return False

    os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)

    logged_names = _load_logged_names()
    if name in logged_names:
        return False

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%H:%M:%S")
        writer.writerow([name, timestamp])

    logged_names.add(name)
    return True
