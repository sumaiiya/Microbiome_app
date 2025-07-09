from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import uuid

app = Flask(__name__)
CORS(app)

DATABASE = 'microbiomeTest.sqlite3'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def uid():
    return str(uuid.uuid4())

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def safe_float(val, default=0.0):
    # Helper function to safely convert values to float, fallback to default if invalid or missing
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

@app.route("/api/save_microbiome", methods=["POST"])
def save_microbiome():
    data = request.get_json()
    if not data:
        # Return error if JSON is invalid or missing
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400
    print("Received microbiome data:", data)
    species_list = data.get("species", [])
    #metabolites_list = data.get("metabolites", [])
    #media_list = data.get("media", [])
    # Automatically gather all metabolite IDs from feeding terms
    metabolite_ids = set()

    for sp in species_list:
        for sub in sp.get("subpopulations", []):
            for term in sub.get("feeding", []):
                for item in term.get("in", []):
                    metabolite_ids.add(item.get("metabolite", ""))
                for item in term.get("out", []):
                    metabolite_ids.add(item.get("metabolite", ""))


    conn = get_db_connection()
    cur = conn.cursor()
    term_counter = 1
    try:
        # Loop through each species sent from frontend
        for sp in species_list:
            sp_id =  sp.get('id') # Generate new unique ID for species (UUID)
            cur.execute(
                # Insert species into DB; id, name, color
                "INSERT OR REPLACE INTO species (id, name, color) VALUES (?, ?, ?)",
                (sp['id'], sp['name'], sp['color'])
            )

            # Loop through subpopulations for this species
            for sub in sp.get("subpopulations", []):
                sub_id = sub.get('id')  # Generate unique ID for subpopulation
                cur.execute(
                    """INSERT OR REPLACE INTO subpopulations (id, species, mumax, pHoptimal, pHalpha, count, color, state)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sub_id,
                        sp_id,  # Foreign key to species ID, NOT species name
                        safe_float(sub.get("mumax"), 0.4),  # Convert to float, default 0.4 if missing/invalid
                        safe_float(sub.get("pHopt"), 7.0),  
                        safe_float(sub.get("pHalpha"), 0.2),  
                        safe_float(sub.get("count"), 0),  
                        sp['color'],  
                        sub.get("state", "active")
                    )
                )
               
                # Now process feeding terms associated with this subpopulation
                for term in sub.get("feeding", []):
                    term_id = term.get('name')
                   

                    cur.execute(
                        # Insert feedingTerm record with id, name, and species ID (foreign key)
                        "INSERT OR REPLACE INTO feedingTerms (id, name, species) VALUES (?, ?, ?)",
                        (term_id,term_id, sp_id)  
                    )

                    # Insert 'in' metabolites for feedingTerm
                    for item in term.get("in", []):
                        cur.execute(
                            """INSERT OR REPLACE INTO feedingTerms2metabolites (feedingTerm, metabolite, yield, monodK)
                               VALUES (?, ?, ?, ?)""",
                            (
                                term_id,
                                item.get("metabolite",""),  # metabolite id
                                safe_float(item.get("yield")),  # yield as float, safe conversion
                                safe_float(item.get("monodK"))  # monodK as float, safe conversion
                            )
                        )

                    # Insert 'out' metabolites for feedingTerm
                    for item in term.get("out", []):
                        cur.execute(
                            """INSERT OR REPLACE INTO feedingTerms2metabolites (feedingTerm, metabolite, yield, monodK)
                               VALUES (?, ?, ?, ?)""",
                            (
                                term_id,
                                item.get("metabolite",""),
                                safe_float(item.get("yield")),
                                safe_float(item.get("monodK"))
                            )
                        )

       # Insert metabolites with default color and MolecularWeight
        for met_id in metabolite_ids:
            if not met_id:
                continue  # Skip empty ids
            cur.execute(
                """
                INSERT OR IGNORE INTO metabolites (id, color, MolecularWeight)
                VALUES (?, ?, ?)
                """,
                (
                    met_id,
                    "#000000",  # Default color
                    0           # Default MolecularWeight
                )
            )

        # Insert into media with default concentration 0
        for met_id in metabolite_ids:
            cur.execute(
                "INSERT OR IGNORE INTO kombucha_media (metabolite, concentration) VALUES (?, ?)",
                (
                    met_id,
                    0.0
                )
            )


      

        conn.commit()  # Commit all changes at once
        return jsonify({"status": "success", "message": "Data saved to database."})

    except Exception as e:
        # Rollback if anything goes wrong to keep DB consistent
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Always close DB connection
        conn.close()
# 🔍 API to get all media metabolites and their concentrations
@app.route("/api/get_media", methods=["GET"])
def get_media():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT metabolite, concentration FROM kombucha_media")
    rows = cur.fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


# 🔍 API to get all subpopulations and their counts
@app.route("/api/get_subpopulations", methods=["GET"])
def get_subpopulations():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, count FROM subpopulations")
    rows = cur.fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])


# 🔧 API to update media concentrations and subpopulation counts
@app.route("/api/update_reactor", methods=["POST"])
def update_reactor():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    media_updates = data.get("media", [])
    subpop_updates = data.get("subpopulations", [])

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Update media concentrations
        for item in media_updates:
            cur.execute(
                "UPDATE kombucha_media SET concentration = ? WHERE metabolite = ?",
                (safe_float(item.get("concentration"), 0), item.get("metabolite", ""))
            )

        # Update subpopulation counts
        for sub in subpop_updates:
            cur.execute(
                "UPDATE subpopulations SET count = ? WHERE id = ?",
                (safe_float(sub.get("count"), 0), sub.get("id", ""))
            )

        conn.commit()
        return jsonify({"status": "success", "message": "Reactor updated."})

    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        conn.close()
@app.route("/api/clear_microbiome", methods=["POST"])
def clear_microbiome():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM feedingTerms2metabolites")
        cur.execute("DELETE FROM feedingTerms")
        cur.execute("DELETE FROM subpopulations")
        cur.execute("DELETE FROM species")
        cur.execute("DELETE FROM metabolites")
        cur.execute("DELETE FROM kombucha_media")

        conn.commit()
        return jsonify({"status": "success", "message": "Database cleared."})

    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        conn.close()



if __name__ == "__main__":
    app.run(debug=True)
