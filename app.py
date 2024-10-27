import sqlite3
from datetime import datetime

from flask import Flask, render_template, request
from openai import OpenAI

from db import init_db
from gpt import pipeline, timeline

init_db()
app = Flask(__name__)

client = OpenAI()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/")
@app.route("/add_test_result", methods=["POST"])
def add_test_result():
    user_id = request.form["user_id"]
    inp = request.form["result"]
    date_object = datetime.strptime(request.form["time"], "%Y-%m-%d").date()

    result = pipeline(client, inp)

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO test_results (user_id, result, time) VALUES (?, ?, ?)",
        (user_id, result, date_object),
    )
    conn.commit()

    cursor.execute(
        "SELECT result FROM test_results WHERE user_id = ? ORDER BY time DESC, id DESC",
        (user_id,),
    )
    user_history = cursor.fetchall()

    if len(user_history) > 1:
        combined = " ".join([str(uh) for uh in user_history[:5][::-1]])
        timeline_text = timeline(client, combined)
        cursor.execute(
            "INSERT INTO status (user_id, text) VALUES (?, ?)",
            (user_id, timeline_text),
        )
        conn.commit()
    conn.close()
    return history(user_id)


@app.route("/get_user", methods=["GET"])
def get_user(username=None):
    if not username:
        username = request.args.get("q")
        if not username:
            return "Please enter a valid username.", 400

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user:
        user_id = user[0]
        cursor.execute(
            "SELECT result FROM test_results WHERE user_id = ? ORDER BY time DESC",
            (user_id,),
        )

    else:
        cursor.execute("INSERT INTO users (username) VALUES (?)", (username,))
        conn.commit()
        user_id = cursor.lastrowid

    conn.close()
    reset_button = f"""<div style="margin-bottom: 20px;"><b>Welcome {username}.</b> <br>Refresh the page to logout. <br></div>"""
    history_html = history(user_id)
    new_test_form_html = f"""
    <form hx-post="/add_test_result" hx-target="#user_history" hx-swap="innerHTML" hx-on="htmx:afterRequest: this.reset();">
        <input type="hidden" name="q" id="username" value="{username}">
        <input type="hidden" name="user_id" value="{user_id}">
        <label for="result">How do you feel? &nbsp</label>
        <input type="text" name="result" id="result" required>
        <input type="date" name="time" id="start_date" value="2024-10-27"/>
        <button type="submit">
            Submit
        </button>
        <img class="htmx-indicator" src="static/grid.svg" size="10px" style="width: 20px; height: 20px;">
        <input type="reset" value="Reset" />
    </form>
    """

    return reset_button + new_test_form_html + history_html


def parse_res(res: str):
    tup = eval(res)
    pol = tup[0]
    color = "green" if pol == "Positive" else "red" if pol == "Negative" else "black"
    concern = ", ".join(tup[1])
    cat = ", ".join(tup[2])
    intensity = tup[3]

    return f"<b style='color:{color};'>{pol}</b><br>Concern: {concern}<br>Categories: {cat}<br>Intensity: {intensity}"


def history(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT text FROM status WHERE user_id = ? ORDER BY id DESC", (user_id,)
    )
    status = cursor.fetchall()

    cursor.execute(
        "SELECT result, time FROM test_results WHERE user_id = ? ORDER BY time DESC, id DESC",
        (user_id,),
    )
    user_history = cursor.fetchall()
    conn.close()
    return (
        "<div id='user_history'>"
        "Timeline: "
        + "<b>"
        + (str(status[0][0]) if status else "Not enough tests to create a timeline.")
        + "</b><h3>Test history</h3><ul>"
        + "".join(
            f"<li>{parse_res(result[0])} <br><i>{result[1]}</i></li>"
            for result in user_history
        )
        + "</ul></div>"
        if user_history
        else "<div id='user_history'><p>No past history available.</p></div>"
    )


if __name__ == "__main__":
    app.run(port=8123, debug=True)
