from flask import Flask,request,render_template,jsonify
from src.pipeline.data_predict_pipeline import Predict_Pipeline,InputData




application=Flask(__name__)

app=application

@app.route('/')

def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])

def predict_datapoints():
    if request.method=='GET':
        return render_template('form.html')
    
    else:

        data=InputData(
            carat=float(request.form.get('carat')),
            depth=float(request.form.get('depth')),
            table=float(request.form.get('table')),
            x=float(request.form.get('x')),
            y=float(request.form.get('y')),
            z=float(request.form.get('x')),
            cut=request.form.get('cut'),
            color=request.form.get('color'),
            clarity=request.form.get('clarity')


        )

        final_data=data.Get_dataframe()
        prict_pipeline=Predict_Pipeline()
        pred=prict_pipeline.Predict(final_data)

        result= round(pred[0],2)
        return render_template('form.html',final_result=result)
    


if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)