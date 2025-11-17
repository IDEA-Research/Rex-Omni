import os
from flask import Flask, render_template, request, redirect, url_for, flash


UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'EZEc6*AASw&&Zx'


@app.route("/", methods=['GET', 'POST'])
def index():
    filename = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('Niciun fișier selectat')
            return redirect(request.url)
        
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            flash('Imaginea a fost încărcată cu succes!')
            
            return render_template('index.html', filename=filename)
        else:
            return redirect(request.url)

        
    return render_template('index.html', filename=None)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)