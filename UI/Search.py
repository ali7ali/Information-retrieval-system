from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__, static_url_path='/templates', static_folder="templates")
import main

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        return redirect(url_for('result', dataset=request.form['dataset'], query=request.form['query'] ))

@app.route('/result/<dataset>/<query>')
def result(dataset,query):
    return render_template('result.html', result=main.query_documents(dataset, query))
    #return 'Searching for %s results' % search_for


if __name__ == "__main__":
    app.run(debug=True)
