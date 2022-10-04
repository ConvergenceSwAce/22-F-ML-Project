import os

path = os.getcwd()
warnPath = path+"/warn.txt"
notiPath = path+"/notification.txt"

def notiTextRead():
    with open(notiPath, 'r', encoding='utf8') as f:
        text = f.read()
    return text

def warnTextRead():
    with open(warnPath, 'r', encoding='utf8') as f:
        text = f.read()
    return text

def isEmptyWarn():
    status = os.stat(warnPath).st_size == 0
    return status