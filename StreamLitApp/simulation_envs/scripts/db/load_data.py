import sqlite3
import os
import chardet
import sys

def create_connection(db_file):
    """Create a database connection to the SQLite database"""
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to database: {db_file}")
        return conn
    except sqlite3.Error as e:
        print(e)
        return None

def getTypes(conn, tableName):
    """Get column types of a table"""
    q = f"PRAGMA table_info({tableName});"
    cur = conn.cursor()
    cur.execute(q)
    return cur.fetchall()

def insert2db(conn, tableName, parameterIDs, *params):
    q = ['?'] * len(params)
    tps = getTypes(conn, tableName)
    values2add = []

    for i, v in enumerate(params):
        if v == '':
            values2add.append(None)
            continue
        if tps[i][2] == 'REAL':
            values2add.append(float(v))
        elif tps[i][2] == 'INTEGER':
            values2add.append(int(v))
        else:
            values2add.append(v)

    query = f"INSERT INTO {tableName} ({','.join(parameterIDs)}) VALUES ({','.join(q)})"
    cur = conn.cursor()
    try:
        cur.execute(query, values2add)
        conn.commit()
    except Exception as e:
        print(f"Insert error in table {tableName}: {e}")
        print("Query:", query)
        print("Values:", values2add)

def detect_encoding(file_path):
    """Automatically detect file encoding"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def addFile(conn, filePath, tableName):
    encoding = detect_encoding(filePath)
    print(f"Loading file: {filePath} into table: {tableName} (Encoding: {encoding})")

    with open(filePath, encoding=encoding) as f:
        # Read header
        header_line = ''
        while not header_line.strip():  # skip empty lines
            header_line = f.readline()
        params = header_line.strip().split('\t')

        for line_number, line in enumerate(f, start=2):
            if not line.strip():
                continue  # skip empty lines

            a = line.strip().split('\t')

            if len(a) != len(params):
                print(f"[Line {line_number}] Skipping malformed line in {filePath}: {line.strip()}")
                continue

            try:
                insert2db(conn, tableName, params, *a)
            except Exception as e:
                print(f"[Line {line_number}] Error inserting line: {a}")
                print(e)

def main():
    # Get the database file path from the command-line argument
    if len(sys.argv) != 3:
        print("Usage: python load_data.py <db_file_path> <table_paths>")
        return

    db_file = sys.argv[1]
    
    base_folder = sys.argv[2]#"/content/simulation_envs/files/db_tables"
    conn = create_connection(db_file)
    if conn is None:
        return

    # Files and corresponding tables
    table_files = {
        "elements.tsv": "elements",
        "metabolites.tsv": "metabolites",
        "metabolites2elements.tsv": "metabolites2elements",
        "kombucha_media.tsv": "kombucha_media",
        "species.tsv": "species",
        "feedingTerms.tsv": "feedingTerms",
        "feedingTerms2metabolites.tsv": "feedingTerms2metabolites",
        "subpopulations.tsv": "subpopulations",
        "subpopulations2subpopulations.tsv": "subpopulations2subpopulations",
        "subpopulations2feedingTerms.tsv": "subpopulations2feedingTerms"
    }

    for file_name, table_name in table_files.items():
        file_path = os.path.join(base_folder, file_name)
        if os.path.exists(file_path):
            addFile(conn, file_path, table_name)
        else:
            print(f"File not found: {file_path}")

    conn.close()

if __name__ == "__main__":
    main()
