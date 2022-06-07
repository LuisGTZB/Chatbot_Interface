import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


Alumnos = [{'Nombre':'Luis','Nua':345805, 'Correo': 'jl.gutierrezbecerra@ugto.mx'}, {
        'Nombre':'Eduardo','Nua':345806, 'Correo': 'le.santoyoparamo@ugto.mx'},{ 
        'Nombre':'Roman', 'Nua':345807, 'Correo':'br.lopezcano@ugto.mx'},{ 
        'Nombre':'Adrian', 'Nua':345808, 'Correo':'ad.lopezgarcia@ugto.mx'},{ 
        'Nombre':'Mariana', 'Nua':345809, 'Correo':'me.garciahernandez@ugto.mx'},{ 
        'Nombre':'Jimena', 'Nua':345810, 'Correo':'pj.renteriamondelo@ugto.mx'}]

Empleados = [{'Nombre':'Enrique','Nue':475801, 'Correo': 'ae.martinezhernandez@ugto.mx'},{
        'Nombre':'Jorge','Nue':475802, 'Correo': 'ja.torresmejia@ugto.mx'},{
        'Nombre':'Monica','Nue':475803, 'Correo': 'mm.villasenior@ugto.mx'},{
        'Nombre':'Martha','Nue':475804, 'Correo': 'mp.andradevillas@ugto.mx'},{
        'Nombre':'Andrea','Nue':475805, 'Correo': 'ma.diazmedrano@ugto.mx'},{
        'Nombre':'Daniel','Nue':475806, 'Correo': 'da.gutierrezvalderrama@ugto.mx'}]

def get_user_email(user_profile, user_number): 
    
    if user_profile == "empleado":
        for  Empleado in Empleados:
            if user_number == Empleado['Nue']:
                return "Su correo es:\t" +  Empleado['Correo']  
            else:
                return "Datos erroneos, verifica e intentalo de nuevo"
                
    elif user_profile == "alumno":
        for Alumno in Alumnos:
            if user_number == Alumno['Nua']:
                return "Su correo es:\t" + Alumno['Correo']
            else:
                return "Datos erroneos, verifica e intentalo de nuevo"
 
def get_password(user_profile, user_id):
    if user_profile == "empleado":
        for Empleado in Empleados:
            if user_id == Empleado['Nue']:
                return send_password_email(Empleado['Correo'])
    if user_profile == "alumno":
        for Alumno in Alumnos:
            if user_id == Alumno['Nua']:
                return send_password_email(Alumno['Correo'])
               
#Funcion para enviar contraseña temporal por correo
def send_password_email(user_email):
    return "Se envio una contraseña temporal a su correo registrado: " + user_email

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Disculpa, no te entiendo"

#pruebas en terminal
if __name__ == "__main__":
   
    while True:
        sentence = input("tu: ")
        if sentence == "salir":
            break
            
        resp = get_response(sentence)
        print(resp)