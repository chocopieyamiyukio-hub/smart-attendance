with open('tests/test_database.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('Afternoon 1,S-001,Ada Lovelace', '13:00to14:00,S-001,Ada Lovelace')
with open('tests/test_database.py', 'w', encoding='utf-8') as f:
    f.write(code)

with open('tests/test_reset_data.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace('# removed', 'import database; database.init_db()')
with open('tests/test_reset_data.py', 'w', encoding='utf-8') as f:
    f.write(code)
