import sqlite3

conn = sqlite3.connect("database/attendance.db")

print("\n===== Students =====")
for row in conn.execute("SELECT * FROM Students"):
    print(row)

print("\n===== Attendance =====")
for row in conn.execute("SELECT * FROM Attendance"):
    print(row)

conn.close()