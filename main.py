from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi import Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


model = keras.models.load_model('model.keras')
iris = load_iris()
scaler = StandardScaler()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_test)
def predict(sl,sw,pl,pw):
    input_data = np.array([[sl, sw, pl, pw]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    response_html=str(f"""<!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>6ЛР. Шеламов К203с9-1</title>
        </head>
        <body>
            <p>Длина чашелистика:{sl} </p>
            <p>Ширина чашелистика:{sw}  </p>
            <p>Длина лепестка:{pl} </p>
            <p>Ширина лепестка:{pw} </p>
            <p>Прогнозируемый класс: {iris.target_names[predicted_class][0]}</p>
            <p><a href= "../" >Back</a></p>
        </body>
        </html>""")
    return response_html
    
app = FastAPI()

html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>6ЛР. Шеламов К203с9-1</title>
</head>
<body>
    <h1>Поле ввода значений для ирисов</h1>
    <form action="/submit" method="post">
        <label for="sepal_length">Длина чашелистика:</label><br>
        <input type="number" step="0.01" min="0" id="sepal_length" name="sepal_length" required><br><br>
        
        <label for="sepal_width">Ширина чашелистика:</label><br>
        <input type="number" step="0.01" min="0" id="sepal_width" name="sepal_width" required><br><br>
        
        <label for="petal_length">Длина лепестка:</label><br>
        <input type="number" step="0.01" min="0" id="petal_length" name="petal_length" required><br><br>
        
        <label for="petal_width">Ширина лепестка:</label><br>
        <input type="number" step="0.01" min="0" id="petal_width" name="petal_width" required><br><br>

        <button type="submit">Отправить</button>
    </form>
</body>
</html>
"""
#request: Request)
@app.get("/", response_class=HTMLResponse)
async def read_form():
    return html_content

@app.post("/submit", response_class=HTMLResponse)
async def handle_form(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
    
#

):
    html_res_content=predict(sepal_length,sepal_width,petal_length,petal_width)
    return HTMLResponse(content=html_res_content, status_code=200) 
    