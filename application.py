import flask
from flask import request, jsonify
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import json

app = flask.Flask(__name__)
df = pd.read_csv("train_data1.csv")
df_X = df.iloc[:, 1:12].copy()  # Train Input
df_Y = df.iloc[:, 12:16].copy()  # Train Output
vendorArray = []
forest = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest_updated = MultiOutputClassifier(forest, n_jobs=-1)
multi_target_forest.fit(df_X, df_Y)

df = pd.read_csv("train_data2.csv")
df_X = df.iloc[:, 1:12].copy()  # Train Input
df_Y = df.iloc[:, 12:16].copy()  # Train Output
# coursesArray = []
forest1 = RandomForestClassifier(n_estimators=100, random_state=1)
multi_target_forest1 = MultiOutputClassifier(forest1, n_jobs=-1)
multi_target_forest_updated1 = MultiOutputClassifier(forest1, n_jobs=-1)
multi_target_forest1.fit(df_X, df_Y)

def findVendor(index):
    vendors = {
        0: "ACE American Insurance Company",
        1: "American Agri-Business Insurance Company",
        2: "Hudson Insurance Company",
        3: "Crop Risk Services, Inc",
        4: "Motzz Laboratory Inc",
        5: "Agrolab Inc",
        6: "Brookside Laboratories Inc",
        7: "Rock River Laboratory Inc",
        8: "Ashoka Innovators for the Public",
        9: "Institute for Agriculture and Trade Policy (IATP)",
        10: "One Acre Fund",
        11: "Ecoagriculture Partners",
        12: "Red Table Grape",
        13: "Table Grapes from Italy",
        14: "Fresh Seedless Grapes of Indian Origin",
        15: "Sultani Grapes from Turkey"
    }
    return vendors.get(index, "NA")
    # coursesArray.append(courses.get(index, "NA"))

def getVendor():
    df = pd.read_csv("data.csv")
    result = multi_target_forest.predict(df)
    for x in result:
        for i, y in enumerate(x):
            if (y == 1):
                return findVendor(i)

def checkLoan():
    df = pd.read_csv("data1.csv")
    result = multi_target_forest_updated1.predict(df.iloc[:, 1:12].copy())
    for x in result:
        for i, y in enumerate(x):
            if (y == 1):
                return True
            else:
                return False

@app.route('/')
def hello_world():
    return 'Hey its Python Flask application!!'

def trainUpdatedData():
    df = pd.read_csv("updatedata.csv")
    df_a = df.iloc[:, 1:12].copy()  # Train Input
    df_b = df.iloc[:, 12:16].copy()  # Train Output
    multi_target_forest_updated.fit(df_a, df_b)

@app.route('/getVendors', methods=['POST'])
def predictCourse():
    response = []
    for applicationData in json.loads(request.data):
        id = applicationData["Id"]
        x = getVendors(applicationData)
        response.append({"Id": id, "Vendors": x})
    return jsonify(response)


def getVendors(applicationData):
    vendorArray.clear()
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
            predictedCourses = getVendor()
            return (predictedCourses)


@app.route('/checkCrop', methods=['POST'])
def predictUpdatedCourse():
    vendorArray.clear()
    response = []
    for applicationData in json.loads(request.data):
        id = applicationData["Id"]
        with open("data1.csv", 'w') as resultFile:
            resultFile.truncate()
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerow(applicationData.keys())
            wr.writerow(applicationData.values())
            resultFile.close()
        x = checkLoan()
        response.append({"Id": id, "Checked": x})
    return jsonify(response)

if __name__ == '__main__':
    app.debug = True
    app.run(use_reloader=False)
