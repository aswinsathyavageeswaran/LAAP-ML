import flask
from flask import request, jsonify
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import json

app = flask.Flask(__name__)
df = pd.read_csv("train_data.csv")
df_X = df.iloc[:, 1:12].copy()  # Train Input
df_Y = df.iloc[:, 12:16].copy()  # Train Output
coursesArray = []
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest_updated = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(df_X, df_Y)

def findCourse(index):
    courses = {
        0: "Fully Complied",
        1: "Partially Complied",
        2: "Need Further Scrutiny",
        3: "Not Complied"
    }
    return courses.get(index, "NA")
    # coursesArray.append(courses.get(index, "NA"))

def getCourse():
    df = pd.read_csv("data.csv")
    result = multi_target_forest.predict(df)
    for x in result:
        for i, y in enumerate(x):
            if (y == 1):
                return findCourse(i)

def getUpdatedCourse():
    df = pd.read_csv("data.csv")
    result = multi_target_forest_updated.predict(df.iloc[:, 1:12].copy())
    for x in result:
        for i, y in enumerate(x):
            if (y == 1):
                return findCourse(i)
    # return coursesArray

@app.route('/')
def hello_world():
    return 'Hey its Python Flask application!!'

def trainUpdatedData():
    df = pd.read_csv("updatedata.csv")
    df_a = df.iloc[:, 1:12].copy()  # Train Input
    df_b = df.iloc[:, 12:16].copy()  # Train Output
    multi_target_forest_updated.fit(df_a, df_b)

@app.route('/checkCompliance', methods=['POST'])
def predictCourse():
    response = []
    for applicationData in json.loads(request.data):
        id = applicationData["Id"]
        x = predictCompliance(applicationData)
        response.append({"Id": id, "Prediction": x})
    return jsonify(response)


def predictCompliance(applicationData):
    coursesArray.clear()
    del applicationData["Id"]
    for data in applicationData.keys():
        if isinstance(applicationData[data], bool):
            if (applicationData[data]):
                applicationData[data] = 1
            else:
                applicationData[data] = 0
        del applicationData["UpdateData"]
        del applicationData["IsDataEdited"]

        with open("data.csv", 'w') as resultFile:
            resultFile.truncate()
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(applicationData.keys())
            wr.writerow(applicationData.values())
            resultFile.close()
            predictedCourses = getCourse()
            return (predictedCourses)


@app.route('/checkUpdatedCompliance', methods=['POST'])
def predictUpdatedCourse():
    coursesArray.clear()
    applicationData =  json.loads(request.data)
    id = applicationData["Id"]
    response = []
    if applicationData["UpdateData"] == 1:
        del applicationData["UpdateData"]
        del applicationData["IsDataEdited"]
        with open("updatedata.csv", 'w') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(applicationData.keys())
            wr.writerow(applicationData.values())
            # resultFile.close()
            response.append({"UpdatedTheModel": 1})
            return jsonify(response)
    else:
        if applicationData["IsDataEdited"] == 1:
            # del applicationData["Id"]
            trainUpdatedData()
            del applicationData["UpdateData"]
            del applicationData["IsDataEdited"]
            with open("data.csv", 'w') as resultFile:
                wr = csv.writer(resultFile, dialect='excel')
                wr.writerow(applicationData.keys())
                wr.writerow(applicationData.values())
                resultFile.close()
                predictedCourses = getUpdatedCourse()
                response.append({"Id": id, "Prediction": predictedCourses})
                return jsonify(response)

if __name__ == '__main__':
    app.debug = True
    app.run(use_reloader=False)
