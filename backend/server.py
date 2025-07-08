from flask import Flask, request, jsonify
import sqlite3
import uuid

app = Flask(__name__)
DATABASE = 'microbiome_new.sqlite3'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def uid():
    return str(uuid.uuid4())

@app.route("/api/save_microbiome", methods=["POST"])
def save_microbiome():
    data = request.get_json()
    print("Received data:", data)

    species_list = data.get("species", [])
    feeding_terms = data.get("feedingTerms", [])

    conn = get_db_connection()
    cur = conn.cursor()

    for sp in species_list:
        sp_id = uid()
        cur.execute(
            "INSERT INTO species (id, name, color) VALUES (?, ?, ?)",
            (sp_id, sp['name'], sp['color'])
        )
        for sub in sp.get("subpopulations", []):
            sub_id = uid()
            cur.execute(
                """INSERT INTO subpopulations (id, species, mumax, pHoptimal, pHalpha, count, color, state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sub_id,
                    sp_id,
                    float(sub.get("mumax", 0.4)),
                    float(sub.get("pHopt", 5.5)),
                    float(sub.get("pHalpha", 0.2)),
                    float(sub.get("count", 1e6)),
                    sp['color'],
                    "active"
                )
            )

    for term in feeding_terms:
        term_id = uid()
        cur.execute(
            "INSERT INTO feedingTerms (id, name, species) VALUES (?, ?, ?)",
            (term_id, term["name"], species_list[0]["name"] if species_list else "Unknown")
        )

        for item in term.get("in", []):
            cur.execute(
                """INSERT INTO feedingTerms2metabolites (feedingTerm, metabolite, yield, monodK)
                   VALUES (?, ?, ?, ?)""",
                (term_id, item["name"], 1.0, 0.5)
            )

        for item in term.get("out", []):
            cur.execute(
                """INSERT INTO feedingTerms2metabolites (feedingTerm, metabolite, yield, monodK)
                   VALUES (?, ?, ?, ?)""",
                (term_id, item["name"], 1.0, 0.5)
            )

    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Data saved to database."})


if __name__ == "__main__":
    app.run(debug=True)
