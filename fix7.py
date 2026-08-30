with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('import sqlite3; conn=sqlite3.connect(db_path); conn.execute("CREATE TABLE Attendance(id integer)"); conn.execute("CREATE TABLE Students(id integer)"); conn.close()', 'import database; database.DATABASE_PATH = db_path; database.init_db()')
with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
    f.write(code)
