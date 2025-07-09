# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import sqlite3

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_db.py <database_name>")
        sys.exit(1)

    database_name = sys.argv[1]

    # Define the default folder to store the sqlite3 file
    #database_folder = Path(__file__).resolve().parents[2] / 'files' / 'dbs'
    #database_folder.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist

    #database_path = database_folder / database_name
    print(database_name)
    con = sqlite3.connect(database_name)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY NOT NULL UNIQUE,
        name TEXT NOT NULL UNIQUE,
        MolecularWeight REAL NOT NULL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metabolites (
        id TEXT PRIMARY KEY NOT NULL UNIQUE,
        color TEXT NOT NULL,
        MolecularWeight REAL NOT NULL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS metabolites2elements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metabolite TEXT NOT NULL,
        element TEXT NOT NULL,
        atoms INTEGER NOT NULL,
        FOREIGN KEY (metabolite) REFERENCES metabolites (id),
        FOREIGN KEY (element) REFERENCES elements (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS kombucha_media (
        metabolite TEXT PRIMARY KEY UNIQUE,
        concentration REAL NOT NULL,
        FOREIGN KEY (metabolite) REFERENCES metabolites (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS species (
        id TEXT PRIMARY KEY UNIQUE,
        name TEXT NOT NULL,
        genomeSize INTEGER,
        geneNumber INTEGER,
        patricID TEXT,
        ncbiID TEXT,
	color
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedingTerms (
        id TEXT PRIMARY KEY UNIQUE,
        name TEXT NOT NULL,
        species TEXT NOT NULL,
        FOREIGN KEY (species) REFERENCES species (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedingTerms2metabolites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feedingTerm TEXT NOT NULL,
        metabolite TEXT NOT NULL,
        yield REAL NOT NULL,
        monodK REAL NOT NULL,
        FOREIGN KEY (feedingTerm) REFERENCES feedingTerms (id),
        FOREIGN KEY (metabolite) REFERENCES metabolites (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subpopulations (
        id TEXT PRIMARY KEY UNIQUE,
        species TEXT NOT NULL,
        mumax REAL NOT NULL,
        pHoptimal REAL NOT NULL,
        pHalpha REAL NOT NULL,
        count REAL NOT NULL,
        color TEXT NOT NULL,
        state TEXT NOT NULL,
        FOREIGN KEY (species) REFERENCES species (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subpopulations2feedingTerms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subpopulation TEXT NOT NULL,
        feedingTerm TEXT NOT NULL,
        FOREIGN KEY (subpopulation) REFERENCES subpopulations (id),
        FOREIGN KEY (feedingTerm) REFERENCES feedingTerms (id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS subpopulations2subpopulations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subpopulation_A TEXT NOT NULL,
        subpopulation_B TEXT NOT NULL,
        hillFunc TEXT NOT NULL,
        rate REAL NOT NULL,
        FOREIGN KEY (subpopulation_A) REFERENCES subpopulations (id),
        FOREIGN KEY (subpopulation_B) REFERENCES subpopulations (id)
    )""")

    con.commit()
    con.close()
    print(f"Database created successfully at: {database_name}")

if __name__ == "__main__":
    main()
