from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# We wake the pipeline up HERE, outside the route, so it stays in RAM!
predict_pipeline = PredictPipeline()

@app.route('/')
def index():
    return render_template('index.html', form_data=None)

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            Order_Value=float(request.form.get('Order_Value')),
            Delivery_Time_Mins=float(request.form.get('Delivery_Time_Mins')) * 1440,
            Latitude=float(request.form.get('Latitude')),
            Longitude=float(request.form.get('Longitude'))
        )
        
        pred_df = data.get_data_as_data_frame()
        status, warehouse = predict_pipeline.predict(pred_df)
        
        # Notice we are passing 'request.form' back to the HTML!
        return render_template('index.html', status=status, warehouse=warehouse, form_data=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)