# from kaggle.competitions import nflrush
import pandas as pd
import numpy as np
import os
import seaborn as sns
from datetime import date
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
%matplotlib inline


nfl = pd.read_csv("train.csv")
#drop columns that we dont need after analysing

nfl.drop(labels = ['GameId', 'PlayId','NflId', 'DisplayName', 'JerseyNumber', 'Season',
       'NflIdRusher','PlayerCollegeName', 'HomeTeamAbbr', 'VisitorTeamAbbr',
         'Stadium', 'Location', 'TimeHandoff', 'TimeSnap'], inplace = True, axis=1)
##########################################################3
#functions we use to change words to integer so our algorithm can understand (label encode)
def get_count(arr):
    count = {}

    pos = 0
    for j in arr:
        if j not in count:
            count[j]= pos
            pos+=1
    return count



def get_count2(arr):
    count = {}

    pos = 0
    for j in arr:
        if j not in count:
            count[j]= pos
            pos+=0.2
    return count

def get_count3(arr):
    count = {}

    pos = 0
    for j in arr:
        if j not in count:
            count[j]= pos
            pos+=0.5
    return count


def clean(x):
    x = x.replace("-", ".")
    return float(x)

def get_sec(time_str):
    m, s, ss = time_str.split(':')
    return int(m) * 60 + int(s) + int(ss)


def calculate_age(born):
    born = datetime.strptime(born, "%m/%d/%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
#############################################################################
#using the ffill method to fill missing data

nfl.fillna(method ='ffill', inplace = True)
#checking to make sure we dont have missing data 
nfl.isnull().sum()
#still had few columns that didnt respond to the ffil method so i did it individually for each column
nfl["GameWeather"].fillna( method ='ffill', inplace = True)
nfl["Temperature"].fillna( method ='ffill', inplace = True)
nfl["Humidity"].fillna( method ='ffill', inplace = True)
nfl["WindSpeed"].fillna( method ='ffill', inplace = True)
nfl["WindDirection"].fillna( method ='ffill', inplace = True)
nfl["FieldPosition"].fillna( method ='ffill', inplace = True)
nfl["Orientation"].fillna( method ='ffill', inplace = True)
nfl["Dir"].fillna( method ='ffill', inplace = True)
nfl["OffenseFormation"].fillna( method ='ffill', inplace = True)
nfl["DefendersInTheBox"].fillna( method ='ffill', inplace = True)
nfl["WindSpeed"].fillna( method ='ffill', inplace = True)
nfl["StadiumType"].fillna( method ='ffill', inplace = True)

#Label encode
Team = {'home':1, 'away':0}
nfl['Team'] = nfl['Team'].map(Team)

#convert time column to secs 

arr = nfl['GameClock'].unique()
count = {}
for i in arr:
    if i not in count:
        count[i] = get_sec(i)


GameClock = count
nfl['GameClock'] = nfl['GameClock'].map(GameClock)

#PossesionTeam
possessionteam = nfl['PossessionTeam'].unique()
PossesionTeam = get_count(possessionteam)


PossessionTeam = {'NE': 0,'KC': 0.5, 'BUF': 1.0, 'NYJ': 1.5, 'ATL': 2.0, 'CHI': 2.5, 'CIN': 3.0, 'BLT': 3.5, 'CLV': 4.0,
 'PIT': 4.5, 'ARZ': 5.0, 'DET': 5.5, 'JAX': 6.0, 'HST': 6.5, 'OAK': 7.0, 'TEN': 7.5, 'WAS': 8.0,
 'PHI': 8.5, 'LA': 9.0, 'IND': 9.5, 'SEA': 10.0, 'GB': 10.5, 'CAR': 11.0, 'SF': 11.5, 'DAL': 12.0,
 'NYG': 12.5, 'NO': 13.0, 'MIN': 13.5, 'DEN': 14.0, 'LAC': 14.5, 'TB': 15.0, 'MIA': 15.5}

nfl['PossessionTeam'] = nfl['PossessionTeam'].map(PossessionTeam)
#fieldposition applying label encoding 
fieldposition = nfl['FieldPosition'].unique()
FieldPosition = get_count(fieldposition)
FieldPosition

nfl['FieldPosition'] = nfl['FieldPosition'].map(FieldPosition)

#OffenseFormation
offenseformation = nfl['OffenseFormation'].unique()
OffenseFormation = get_count3(offenseformation)
nfl['OffenseFormation'] = nfl['OffenseFormation'].map(OffenseFormation)

OffenseFormation= {'SHOTGUN': 0,'SINGLEBACK': 0.5,'JUMBO': 1.0, 'PISTOL': 1.5, 'I_FORM': 2.0, 'ACE': 2.5,
 'WILDCAT': 3.0,'EMPTY': 3.5}

#changing defenders in the box to integers
nfl['DefendersInTheBox'] = nfl['DefendersInTheBox'].astype(int)
#play direction
PlayDirection= {'left':0, 'right':1}
nfl['PlayDirection'] = nfl['PlayDirection'].map(PlayDirection)

#cleaning players height
nfl['PlayerHeight'] = nfl['PlayerHeight'].apply(clean)


#changing players data of birth 
nfl['PlayerBirthDate'] = nfl['PlayerBirthDate'].apply(calculate_age)
#Position
position = nfl['Position'].unique()
Position = get_count2(position)
nfl['Position'] = nfl['Position'].map(Position)

Position = {'SS': 0,'DE': 0.2, 'ILB': 0.4, 'FS': 0.6000000000000001, 'CB': 0.8, 'DT': 1.0, 'WR': 1.2,
        'TE': 1.4, 'T': 1.5999999999999999, 'QB': 1.7999999999999998, 'RB': 1.9999999999999998,
 'G': 2.1999999999999997, 'C': 2.4, 'OLB': 2.6, 'NT': 2.8000000000000003, 'FB': 3.0000000000000004,
 'MLB': 3.2000000000000006, 'LB': 3.400000000000001, 'OT': 3.600000000000001, 'OG': 3.800000000000001,
 'HB': 4.000000000000001, 'DB': 4.200000000000001, 'S': 4.400000000000001, 'DL': 4.600000000000001,
 'SAF': 4.800000000000002}

 #changing all mistakes in stadium type to correct format either indoor or outdoor
 StadiumType = {'Outdoor': 'Outdoor',
 'Outdoors': 'Outdoor',
 'Indoors': 'Indoor',
 'Retractable Roof': 'Indoor',
 'Indoor': 'Indoor',
 'Retr. Roof-Closed': 'Indoor',
 'Open': 'Outdoor',
 'Indoor, Open Roof': 'Outdoor',
 'Retr. Roof - Closed': 'Indoor',
 'Outddors': 'Outdoor',
 'Dome': 'Indoor',
 'Domed, closed': 'Indoor',
 'Indoor, Roof Closed': 'Indoor',
 'Retr. Roof Closed': 'Indoor',
 'Outdoor Retr Roof-Open': 'Outdoor',
 'Closed Dome': 'Indoor',
 'Oudoor': 'Outdoor',
 'Ourdoor': 'Outdoor',
 'Dome, closed': 'Indoor',
 'Retr. Roof-Open': 'Outdoor',
 'Heinz Field': 'Outdoor',
 'Outdor': 'Outdoor',
 'Retr. Roof - Open': 'Outdoor',
 'Domed, Open': 'Outdoor',
 'Domed, open': 'Outdoor',
 'Cloudy': 'Outdoor',
 'Bowl': 'Outdoor',
 'Outside': 'Outdoor',
 'Domed': 'Indoor'}
nfl['StadiumType'] = nfl['StadiumType'].map(StadiumType)

StadiumType = get_count(nfl['StadiumType'].unique())
nfl['StadiumType'] = nfl['StadiumType'].map(StadiumType)
#changing all mistakes in turf to correct format 
Turf = {'Field Turf': 'Field Turf',
 'A-Turf Titan': 'A-Turf Titan',
 'Grass': 'Grass',
 'UBU Sports Speed S5-M': 'UBU Sports Speed S5-M',
 'Artificial': 'Artificial',
 'DD GrassMaster': 'Grass',
 'Natural Grass': 'Grass',
 'UBU Speed Series-S5-M': 'UBU Sports Speed S5-M',
 'FieldTurf': 'Field Turf',
 'FieldTurf 360': 'Field Turf',
 'Natural grass': 'Grass',
 'grass': 'Grass',
 'Natural': 'Grass',
 'Artifical': 'Artificial',
 'FieldTurf360': 'Field Turf',
 'Naturall Grass': 'Grass',
 'Field turf': 'Field Turf',
 'SISGrass': 'Grass',
 'Twenty-Four/Seven Turf': 'Twenty-Four/Seven Turf',
 'natural grass': 'Grass'}
nfl['Turf'] = nfl['Turf'].map(Turf)

turf = nfl['Turf'].unique()
Turf = get_count2(turf)
#Turf
nfl['Turf'] = nfl['Turf'].map(Turf)

Turf={'Field Turf': 0,'A-Turf Titan': 0.2, 'Grass': 0.4, 'UBU Sports Speed S5-M': 0.6000000000000001,
      'Artificial': 0.8, 'Twenty-Four/Seven Turf': 1.0}

#changing all mistakes in game weather to correct format 
GameWeather = {'Clear and warm': 'Clear',
 'Sun & clouds': 'Sunny',
 'Sunny': 'Sunny',
 'Controlled Climate': 'Controlled Climate',
 'Mostly Sunny': 'Sunny',
 'Clear': 'Clear',
 'Indoor': 'Controlled Climate',
 'Mostly Cloudy': 'Cloudy',
 'Mostly Coudy': 'Cloudy',
 'Partly sunny': 'Sunny',
 'Partly Cloudy': 'Cloudy',
 'Cloudy': 'Cloudy',
 'Sunny, highs to upper 80s': 'Sunny',
 'Indoors': 'Controlled Climate',
 'Light Rain': 'Rain',
 'Showers': 'Rain',
 'Partly cloudy': 'Cloudy',
 'Partly Sunny': 'Sunny',
 '30% Chance of Rain': 'Rain',
 'Cloudy with periods of rain, thunder possible. Winds shifting to WNW, 10-20 mph.': 'Cloudy',
 'Rain': 'Rain',
 'Cloudy, fog started developing in 2nd quarter': 'Cloudy',
 'Coudy': 'Cloudy',
 'Rain likely, temps in low 40s.': 'Rain',
 'Cold': 'Cold',
 'N/A (Indoors)': 'Controlled Climate',
 'Clear skies': 'Clear',
 'cloudy': 'Cloudy',
 'Fair': 'Cloudy',
 'Mostly cloudy': 'Cloudy',
 'Cloudy, chance of rain': 'Cloudy',
 'Heavy lake effect snow': 'Cold',
 'Party Cloudy': 'Cloudy',
 'Cloudy, light snow accumulating 1-3"': 'Cloudy',
 'Cloudy and cold': 'Cloudy',
 'Snow': 'Cold',
 'Hazy': 'Rain',
 'Scattered Showers': 'Rain',
 'Cloudy and Cool': 'Cloudy',
 'N/A Indoor': 'Controlled Climate',
 'Rain Chance 40%': 'Rain',
 'Clear and sunny': 'Sunny',
 'Mostly sunny': 'Sunny',
 'Sunny and warm': 'Sunny',
 'Partly clear': 'Clear',
 'Cloudy, 50% change of rain': 'Cloudy',
 'Clear and Sunny': 'Sunny',
 'Sunny, Windy': 'Sunny',
 'Clear and Cool': 'Clear',
 'Sunny and clear': 'Sunny',
 'Mostly Sunny Skies': 'Sunny',
 'Partly Clouidy': 'Cloudy',
 'Clear Skies': 'Clear',
 'Sunny Skies': 'Sunny',
 'Overcast': 'Cloudy',
 'T: 51; H: 55; W: NW 10 mph': 'Cloudy',
 'Cloudy, Rain': 'Cloudy',
 'Rain shower': 'Rain',
 'Clear and cold': 'Clear',
 'Rainy': 'Rain',
 'Sunny and cold': 'Sunny'}
nfl['GameWeather'] = nfl['GameWeather'].map(GameWeather)
#GameWeather
gameweather = nfl['GameWeather'].unique()
GameWeather = get_count2(gameweather)
nfl['GameWeather'] = nfl['GameWeather'].map(GameWeather)
GameWeather ={'Clear': 0,'Sunny': 0.2, 'Controlled Climate': 0.4, 'Cloudy': 0.6000000000000001,
 'Rain': 0.8, 'Cold': 1.0}
#changing all mistakes in wind speed to correct format 
 WindSpee =  WindSpee = {8.0: '8', 6.0: '6', 10.0: '10', 9.0: '9', 11.0: '11', 7.0: '7', 5.0: '5', 2.0: '2', 12.0: '12', 1: '1', 3: '3', 4: '4', 13: '13', '10': '10', '5': '5', '6': '6',
      '4':'4','8': '8', '0': '0', 'SSW': '0', 14.0: '14', 0.0: '0', 15.0: '15', 17.0: '17', 18.0: '18', 16.0: '16', '11-17': '14', '16': '16', '14': '14', '13' : '13',  '12': '12',
     '23': '23', '7': '7', '9': '9', '3': '3', '17': '17', '14-23': '19', '1': '1', '13 MPH': '13', 24.0: '24', '15': '15', '12-22': '17', '2': '2', '4 MPh': '4', '15 gusts up to 25': '20',
     '11': '11', '10MPH': '10', '10mph': '10', '22': '22', 'E': '22', '7 MPH': '7', 'Calm': '7', '6 mph': '6', '19': '19', 'SE': '19', '20': '20', '10-20': '20', '12mph': '12'}
nfl['WindSpeed'] = nfl['WindSpeed'].map(WindSpee)

nfl['WindSpeed'] = nfl['WindSpeed'].astype(int)
#changing all mistakes in wind direction to correct format 
Wind_direction = {'SW': 'SW', 'NNE': 'NNE', 'SE': 'SE', 'East': 'East', 'NE': 'NE', 'North': 'North',
 'S': 'S', 'Northwest': 'Northwest', 'SouthWest': 'SW', 'ENE': 'ENE', 'ESE': 'ESE', 'SSW': 'SSW',
 'NW': 'Northwest', 'Northeast': 'NE', 'From S': 'S', 'W': 'W', 'South': 'S', 'West-Southwest': 'WSW', 
                  'E': 'East', '13': 'East', 'N': 'North', 'NNW': 'NNW', 'South Southeast': 'SSE',
 'SSE': 'SSE','West': 'W', 'WSW': 'WSW', 'From SW': 'SW', 'WNW': 'WNW', 's': 'S', 'NorthEast': 'NE',
 'from W': 'W', 'W-NW': 'WNW', 'South Southwest': 'SSW', 'Southeast': 'SE', 'From WSW': 'WSW', 'West Northwest': 'WNW',
 'Calm': 'S', 'From SSE': 'SSE', 'From W': 'W', 'East North East': 'ENE', 'From ESE': 'ESE', 'EAST': 'East',
 'East Southeast': 'ESE', 'From SSW': 'SSW', '8': 'W', 'North East': 'NE',
 'Southwest': 'SW',
 'North/Northwest': 'NNW',
 'From NNE': 'NNE',
 '1': 'N',
 'N-NE': 'NNE',
 'W-SW': 'WSW',
 'From NNW': 'NNW'}

nfl['WindDirection'] = nfl['WindDirection'].map(Wind_direction)
#WindDirection
winddirection = nfl['WindDirection'].unique()
WindDirection = get_count3(winddirection)
nfl['WindDirection'] = nfl['WindDirection'].map(WindDirection)
WindDirection={'SW': 0, 'NNE': 0.5, 'SE': 1.0, 'East': 1.5, 'NE': 2.0, 'North': 2.5, 'S': 3.0,
 'Northwest': 3.5, 'ENE': 4.0, 'ESE': 4.5, 'SSW': 5.0, 'W': 5.5, 'WSW': 6.0,
 'NNW': 6.5, 'SSE': 7.0, 'WNW': 7.5, 'N': 8.0}

#OffensePersonnel
offensepersonnel = nfl['OffensePersonnel'].unique()
OffensePersonnel = get_count3(offensepersonnel)

nfl['OffensePersonnel'] = nfl['OffensePersonnel'].map(OffensePersonnel)

OffensePersonnel = {'1 RB, 1 TE, 3 WR': 0, '6 OL, 2 RB, 2 TE, 0 WR': 0.5, '1 RB, 3 TE, 1 WR': 1.0,
 '1 RB, 2 TE, 2 WR': 1.5, '6 OL, 1 RB, 2 TE, 1 WR': 2.0, '2 RB, 1 TE, 2 WR': 2.5,
 '2 RB, 2 TE, 1 WR': 3.0, '0 RB, 3 TE, 2 WR': 3.5, '0 RB, 1 TE, 4 WR': 4.0, '6 OL, 1 RB, 0 TE, 3 WR': 4.5,
'6 OL, 1 RB, 1 TE, 2 WR': 5.0,'1 RB, 2 TE, 1 WR,1 DL': 5.5, '1 RB, 3 TE, 0 WR,1 DL': 6.0,
 '1 RB, 0 TE, 4 WR': 6.5, '1 RB, 1 TE, 2 WR,1 DL': 7.0, '6 OL, 2 RB, 0 TE, 2 WR': 7.5,
 '2 RB, 0 TE, 3 WR': 8.0, '6 OL, 2 RB, 1 TE, 1 WR': 8.5, '7 OL, 1 RB, 0 TE, 2 WR': 9.0, '7 OL, 2 RB, 0 TE, 1 WR': 9.5,
 '7 OL, 1 RB, 2 TE, 0 WR': 10.0, '2 RB, 3 TE, 0 WR': 10.5, '3 RB, 1 TE, 1 WR': 11.0,
 '6 OL, 1 RB, 3 TE, 0 WR': 11.5, '6 OL, 1 RB, 2 TE, 0 WR,1 DL': 12.0, '2 RB, 3 TE, 1 WR': 12.5,
 '6 OL, 1 RB, 1 TE, 1 WR,1 DL': 13.0, '1 RB, 4 TE, 0 WR': 13.5, '1 RB, 2 TE, 1 WR,1 LB': 14.0,
 '1 RB, 3 TE, 0 WR,1 LB': 14.5, '7 OL, 2 RB, 1 TE, 0 WR': 15.0, '0 RB, 2 TE, 3 WR': 15.5,
 '1 RB, 0 TE, 3 WR,1 DB': 16.0, '6 OL, 1 RB, 2 TE, 0 WR,1 LB': 16.5, '1 RB, 1 TE, 2 WR,1 DB': 17.0,
 '0 RB, 0 TE, 5 WR': 17.5, '1 RB, 2 TE, 3 WR': 18.0, '1 RB, 1 TE, 2 WR,1 LB': 18.5,
 '1 RB, 3 TE, 0 WR,1 DB': 19.0, '6 OL, 2 RB, 1 TE, 0 WR,1 DL': 19.5, '2 QB, 1 RB, 1 TE, 2 WR': 20.0,
 '6 OL, 0 RB, 2 TE, 2 WR': 20.5, '3 RB, 0 TE, 2 WR': 21.0, '2 QB, 1 RB, 2 TE, 1 WR': 21.5,
 '2 QB, 1 RB, 0 TE, 3 WR': 22.0,'3 RB, 2 TE, 0 WR': 22.5,
 '2 RB, 2 TE, 0 WR,1 DL': 23.0, '2 QB, 2 RB, 2 TE, 0 WR': 23.5, '2 QB, 2 RB, 0 TE, 2 WR': 24.0, '2 QB, 3 RB, 1 TE, 0 WR': 24.5,
 '2 QB, 1 RB, 3 TE, 0 WR': 25.0,
 '2 QB, 2 RB, 1 TE, 1 WR': 25.5, '2 RB, 1 TE, 1 WR,1 DB': 26.0, '1 RB, 2 TE, 1 WR,1 DB': 26.5,
 '6 OL, 3 RB, 0 TE, 1 WR': 27.0, '6 OL, 1 RB, 1 TE, 0 WR,2 DL': 27.5}

defensepersonnel = nfl['DefensePersonnel'].unique()
DefensePersonnel = get_count3(defensepersonnel)
#DefensePersonnel
nfl['DefensePersonnel'] = nfl['DefensePersonnel'].map(DefensePersonnel)

DefensePersonnel = {'2 DL, 3 LB, 6 DB': 0, '4 DL, 4 LB, 3 DB': 0.5, '3 DL, 2 LB, 6 DB': 1.0, '3 DL, 4 LB, 4 DB': 1.5,
 '3 DL, 3 LB, 5 DB': 2.0, '4 DL, 3 LB, 4 DB': 2.5, '4 DL, 1 LB, 6 DB': 3.0, '4 DL, 2 LB, 5 DB': 3.5,
 '5 DL, 2 LB, 4 DB': 4.0, '2 DL, 4 LB, 5 DB': 4.5, '2 DL, 5 LB, 4 DB': 5.0, '5 DL, 4 LB, 2 DB': 5.5,
 '1 DL, 5 LB, 5 DB': 6.0, '5 DL, 3 LB, 3 DB': 6.5, '6 DL, 2 LB, 3 DB': 7.0, '3 DL, 5 LB, 3 DB': 7.5,
 '6 DL, 3 LB, 2 DB': 8.0, '1 DL, 3 LB, 7 DB': 8.5, '2 DL, 2 LB, 7 DB': 9.0, '4 DL, 5 LB, 2 DB': 9.5,
 '1 DL, 4 LB, 6 DB': 10.0, '4 DL, 5 LB, 1 DB, 1 OL': 10.5, '6 DL, 1 LB, 4 DB': 11.0, '2 DL, 4 LB, 4 DB, 1 OL': 11.5,
 '6 DL, 4 LB, 1 DB': 12.0, '5 DL, 1 LB, 5 DB': 12.5, '4 DL, 6 LB, 1 DB': 13.0, '0 DL, 5 LB, 6 DB': 13.5,
 '5 DL, 4 LB, 1 DB, 1 OL': 14.0, '3 DL, 1 LB, 7 DB': 14.5, '4 DL, 0 LB, 7 DB': 15.0, '3 DL, 4 LB, 3 DB, 1 OL': 15.5,
 '5 DL, 5 LB, 1 DB': 16.0, '5 DL, 3 LB, 2 DB, 1 OL': 16.5, '0 DL, 6 LB, 5 DB': 17.0, '1 DL, 2 LB, 8 DB': 17.5,
 '0 DL, 4 LB, 7 DB': 18.0, '7 DL, 2 LB, 2 DB': 18.5}
#checking the data type to make sure we have no object colum or string cloum
 nfl.dtypes
#viewing the cleaned data 
nfl.isnull().sum()
nfl.shape
nfl