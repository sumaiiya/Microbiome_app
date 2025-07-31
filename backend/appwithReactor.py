from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import uuid

app = Flask(__name__)
CORS(app)

DATABASE = 'microbiomeOne.sqlite3'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def uid():
    return str(uuid.uuid4())



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
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    print("Received microbiome data:", data)
    model = data.get("model", "")
    setup = data.get("setup", "")  # Treating 'setup' as reactor id
    species_list = data.get("species", [])
    metabolites_list = data.get("metabolites", [])
    media_list = data.get("media", [])

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1. Create reactor entry
        cur.execute(
            "INSERT OR IGNORE INTO reactors (id, model, setup) VALUES (?, ?, ?)",
            (setup, model, setup)
        )

        # 2. Insert metabolites (linked to reactor)
        for m in metabolites_list:
            cur.execute(
                """
                INSERT OR REPLACE INTO metabolites (id, color, MolecularWeight, reactor)
                VALUES (?, ?, ?, ?)
                """,
                (
                    m.get("id", ""),
                    m.get("color", "#000000"),
                    safe_float(m.get("MolecularWeight")),
                    setup
                )
            )

        # 3. Insert media (linked to reactor)
        for m in media_list:
            cur.execute(
                """
                INSERT OR REPLACE INTO kombucha_media (metabolite, concentration, reactor)
                VALUES (?, ?, ?)
                """,
                (
                    m.get("metabolite", ""),
                    safe_float(m.get("concentration")),
                    setup
                )
            )

        # 4. Insert species and subpopulations
        for sp in species_list:
            sp_id = sp.get('id')
            cur.execute(
                """
                INSERT OR REPLACE INTO species (id, name, color, reactor)
                VALUES (?, ?, ?, ?)
                """,
                (sp_id, sp.get('name'), sp.get('color'), setup)
            )

            for sub in sp.get("subpopulations", []):
                sub_id = sub.get('id')
                cur.execute(
                    """
                    INSERT OR REPLACE INTO subpopulations (id, species, mumax, pHoptimal, pHalpha, count, color, state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sub_id,
                        sp_id,
                        safe_float(sub.get("mumax"), 0.4),
                        safe_float(sub.get("pHopt"), 7.0),
                        safe_float(sub.get("pHalpha"), 0.2),
                        safe_float(sub.get("count"), 1e6),
                        sp.get('color', '#888888'),
                        "active"
                    )
                )

                # 5. Feeding terms for this subpopulation
                for term in sub.get("feeding", []):
                    term_id = term.get('name', uid())  # fallback to UUID if no name
                    cur.execute(
                        """
                        INSERT OR REPLACE INTO feedingTerms (id, name, species)
                        VALUES (?, ?, ?)
                        """,
                        (term_id, term_id, sp_id)
                    )

                    # Link subpopulation ↔ feedingTerm
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO subpopulations2feedingTerms (subpopulation, feedingTerm)
                        VALUES (?, ?)
                        """,
                        (sub_id, term_id)
                    )

                    # Insert 'in' metabolites
                    for item in term.get("in", []):
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO feedingTerms2metabolites (feedingTerm, metabolite, reactor, yield, monodK)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                term_id,
                                item.get("metabolite", ""),
                                setup,
                                safe_float(item.get("yield")),
                                safe_float(item.get("monodK"))
                            )
                        )

                    # Insert 'out' metabolites
                    for item in term.get("out", []):
                        cur.execute(
                            """
                            INSERT OR REPLACE INTO feedingTerms2metabolites (feedingTerm, metabolite, reactor, yield, monodK)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                term_id,
                                item.get("metabolite", ""),
                                setup,
                                safe_float(item.get("yield")),
                                safe_float(item.get("monodK"))
                            )
                        )

        conn.commit()
        return jsonify({"status": "success", "message": "Data saved to database."})

    except Exception as e:
        conn.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        conn.close()



if __name__ == "__main__":
    app.run(debug=True)
