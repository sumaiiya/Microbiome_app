from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import uuid

app = Flask(__name__)
CORS(app)

DATABASE = 'microbiome.sqlite3'

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
    metabolites_list = data.get("metabolites", [])
    media_list = data.get("media", [])

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
                        safe_float(sub.get("count"), 1e6),  
                        sp['color'],  
                        "active"  # Default state is active
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

        # Insert metabolites with insert or ignore (avoid duplicates)
        for m in metabolites_list:
            cur.execute(
                """
                INSERT OR IGNORE INTO metabolites (id, color, MolecularWeight)
                VALUES (?, ?, ?)
                """,
                (
                    m.get("id",""),
                    m.get("color","#000000"),
                    safe_float(m.get("MolecularWeight"))
                )
            )

        # Insert media items, insert or ignore for duplicates
        for m in media_list:
            cur.execute(
                "INSERT OR IGNORE INTO kombucha_media (metabolite, concentration) VALUES (?, ?)",
                (
                    m.get("metabolite",""),
                    safe_float(m.get("concentration"))
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


if __name__ == "__main__":
    app.run(debug=True)
