from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

#Se obtiene la respuesta del usuario
@app.get("/")
def index_get():
    return render_template("base.html")

#Se lanza una respuesta del bot
@app.post("/predict")
def predic():
 
    text = request.get_json().get("message")

    response = get_response(text)
    if response == "entra":
        message = {"answer": "¿Eres alumno o empleado?"}
        usr = request.get_json().get("message")
        return jsonify(usr)
    
    message = {"answer": response}

    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
