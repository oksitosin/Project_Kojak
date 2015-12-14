import flask
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it
patients = pd.read_csv("dummy2.csv")
# patients.columns=['Degree_Masters',
# 'GPA',
# 'Quant_GRE_Normal',
# 'Semester_Fall',
# 'Sub_GRE',
# 'Verbal_GRE_Normal',
# 'Writing_GRE',
# 'School_Accepted',
# 'Status_arizonastateuniversity',
# 'Status_bostonuniveristy',
# 'Status_brownuniversity',
# 'Status_buffalosuny',
# 'Status_carnegiemellon',
# 'Status_clemsonuniversity',
# 'Status_coloradouniversityboulder',
# 'Status_columbiauniversity',
# 'Status_cornelluniversity',
# 'Status_dartmouthcollege',
# 'Status_duke',
# 'Status_epfl',
# 'Status_ethzurich',
# 'Status_georgiatech',
# 'Status_harvarduniversity',
# 'Status_illinoisurbanachampaign',
# 'Status_indianauniversitybloomington',
# 'Status_iowasateuniversity',
# 'Status_johnshopkinsuniversity',
# 'Status_kingabdullahuniversityscienceandtechnology',
# 'Status_mcgilluniversity',
# 'Status_michiganstateuniversity',
# 'Status_mit',
# 'Status_newyorkuniversity',
# 'Status_northcarolinastateuniverisy',
# 'Status_northeasternuniversity',
# 'Status_northwesternuniversity',
# 'Status_ohiostateuniversity',
# 'Status_oregonstateuniversity',
# 'Status_oxford',
# 'Status_pennstateuniversity',
# 'Status_pittsburghu',
# 'Status_princeton',
# 'Status_purdueuniversity',
# 'Status_rensselaerpolytechnicinstitute',
# 'Status_riceuniversity',
# 'Status_rutgers',
# 'Status_simonfraseruniversitysfu',
# 'Status_stanforduniversity',
# 'Status_stateuniversitynewyorkstonybrook',
# 'Status_syracuseuniversity',
# 'Status_tamu',
# 'Status_texasuniversityaustin',
# 'Status_tuftsuniversity',
# 'Status_ucberkeley',
# 'Status_ucdavis',
# 'Status_uci',
# 'Status_ucla',
# 'Status_ucr',
# 'Status_ucsandiego',
# 'Status_ucsb',
# 'Status_ucsc',
# 'Status_umassamherst',
# 'Status_universityalberta',
# 'Status_universityarizona',
# 'Status_universitybritishcolumbia',
# 'Status_universitychicago',
# 'Status_universityflorida',
# 'Status_universitymaryland',
# 'Status_universitymichiganannarbor',
# 'Status_universityminnesota',
# 'Status_universitynotredame',
# 'Status_universityrochester',
# 'Status_universitytoronto',
# 'Status_universityutah',
# 'Status_universityvirginia',
# 'Status_universitywashington',
# 'Status_universitywaterloo',
# 'Status_universitywisconsinmadison',
# 'Status_upenn',
# 'Status_usc',
# 'Status_vanderbiltuniversity',
# 'Status_virginiatech',
# 'Status_yale',
# 'Result_A',
# 'Result_I',
# 'Result_O',
# 'Result_U']

X = patients[['Degree_Masters',
'GPA',
'Is_GPA',
'Is_Quant',
'Is_Verbal',
'Is_Writing',
'Quant_GRE_Normal',
'Semester_Fall',
'Sub_GRE',
'Verbal_GRE_Normal',
'Writing_GRE',
'School_arizonastateuniversity',
'School_bostonuniveristy',
'School_brownuniversity',
'School_buffalosuny',
'School_carnegiemellon',
'School_clemsonuniversity',
'School_coloradouniversityboulder',
'School_columbiauniversity',
'School_cornelluniversity',
'School_dartmouthcollege',
'School_duke',
'School_epfl',
'School_ethzurich',
'School_georgiatech',
'School_harvarduniversity',
'School_illinoisurbanachampaign',
'School_indianauniversitybloomington',
'School_iowasateuniversity',
'School_johnshopkinsuniversity',
'School_kingabdullahuniversityscienceandtechnology',
'School_mcgilluniversity',
'School_michiganstateuniversity',
'School_mit',
'School_newyorkuniversity',
'School_northcarolinastateuniverisy',
'School_northeasternuniversity',
'School_northwesternuniversity',
'School_ohiostateuniversity',
'School_oregonstateuniversity',
'School_oxford',
'School_pennstateuniversity',
'School_pittsburghu',
'School_princeton',
'School_purdueuniversity',
'School_rensselaerpolytechnicinstitute',
'School_riceuniversity',
'School_rutgers',
'School_simonfraseruniversitysfu',
'School_stanforduniversity',
'School_stateuniversitynewyorkstonybrook',
'School_syracuseuniversity',
'School_tamu',
'School_texasuniversityaustin',
'School_tuftsuniversity',
'School_ucberkeley',
'School_ucdavis',
'School_uci',
'School_ucla',
'School_ucr',
'School_ucsandiego',
'School_ucsb',
'School_ucsc',
'School_umassamherst',
'School_universityalberta',
'School_universityarizona',
'School_universitybritishcolumbia',
'School_universitychicago',
'School_universityflorida',
'School_universitymaryland',
'School_universitymichiganannarbor',
'School_universityminnesota',
'School_universitynotredame',
'School_universityrochester',
'School_universitytoronto',
'School_universityutah',
'School_universityvirginia',
'School_universitywashington',
'School_universitywaterloo',
'School_universitywisconsinmadison',
'School_upenn',
'School_usc',
'School_vanderbiltuniversity',
'School_virginiatech',
'School_yale',
'Status_A',
'Status_I',
'Status_O',
'Status_U']]
Y = patients['Accepted']
rf = RandomForestClassifier(n_estimators=500, oob_score=True, 
                            criterion='entropy', max_features='log2')
PREDICTOR = rf.fit(X, Y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("awesome.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(debug = True)
