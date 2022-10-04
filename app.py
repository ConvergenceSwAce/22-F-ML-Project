from flask import Flask
from model.readFile import isEmptyWarn, notiTextRead,warnTextRead

app = Flask(__name__)

@app.route('/api/notification')
def notify():
    if(not isEmptyWarn()):
        return warnTextRead()
    return notiTextRead()


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)