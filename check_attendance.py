import sqlite3

conn = sqlite3.connect("database/attendance.db")

print("Students:")
for row in conn.execute("SELECT * FROM Students"):
    print(row)

print("\nAttendance:")
for row in conn.execute("SELECT * FROM Attendance"):
    print(row)

conn.close()