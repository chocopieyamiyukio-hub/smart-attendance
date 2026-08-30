with open('tests/test_database.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('Morning,S-001,Ada Lovelace', '8:45to10:45,S-001,Ada Lovelace')
with open('tests/test_database.py', 'w', encoding='utf-8') as f:
    f.write(code)

with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('db_path.write_bytes(b"old-db")', '# removed')
with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
    f.write(code)
