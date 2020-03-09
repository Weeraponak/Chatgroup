from django.shortcuts import render
from joblib import load

# Create your views here.

def index(req):
    model = load('./chat_group/static/chatgroup.model')
    label = ['']
    chat  = ""
    if req.method == 'POST':
        print("POST IN")
        chat = str(req.POST['chat'])
        label = model.predict([chat])
    return render(req, 'chat_group/index.html' ,{
        'label':label[0],
        'chat':chat,
    })

