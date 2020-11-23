import json
from os import listdir
import re
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import torch
import torch.nn as nn
import os
import hickle as hkl
import numpy as np
import data as d


def normalize(data):
    n = np.mean(data)
    n_ = data - n
    n_n = np.sqrt(np.sum(n_ ** 2))
    n_n_ = n_ / n_n
    return n_n_ * 0.5 + 0.5


def prepare_data(tdata):
    # remove player data
    print(tdata.shape)
    ret_tdata = np.zeros((25, tdata.shape[1]))
    ret_tdata[:5] = tdata[:5]
    ret_tdata[5:26] = tdata[61:]
    return ret_tdata

class FootballData(Dataset):
    """Football Dataset."""

    def __init__(self, data_file, transform=None):
        self.full_data = load_np(data_file)

        norm_list = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        for i in norm_list:
            self.full_data[i] = normalize(self.full_data[i])

        self.tdata = rm_old_games(self.full_data)
        self.odd = np.zeros(3)

        print(self.full_data.shape)
        print(self.tdata.shape)

        self.nb_teams = 23
        self.transform = transform

    def __len__(self):
        return self.tdata.shape[1]

    def __getitem__(self, idx):
        self.home, self.home_rat, self.away, self.away_rat = d.get_train_data(
            self.full_data, date=self.tdata[0, idx], home=self.tdata[1, idx], away=self.tdata[2, idx])
        sample = np.concatenate([self.home, self.away], axis=0)
        sample_rat = np.concatenate([self.home_rat, self.away_rat], axis=0)
        odds = self.tdata[-3:, idx]

        result = self.tdata[9:12, idx]
        # sample_rat = np.concatenate([sample_rat, result], axis=0)

        if self.transform:
            sample = self.transform(sample)

        return sample, np.concatenate([sample_rat, odds], axis=0), result


def rm_old_games(data):
    indices = []
    for i in range(data.shape[-1]):
        try:
            home, home_rat, away, away_rat = d.get_train_data(
                data, date=data[0, i], home=data[1, i], away=data[2, i])
            indices.append(i)
        except IndexError:
            pass
    return data[:, indices]



def save_np(np_array, file_name):
    hkl.dump(np_array, file_name, mode='w', compression='gzip')
    print('Saved File')
    pass

def load_np(file_name):
    return hkl.load(file_name)


def load_understat_data():
    # date: [home away homescore awayscore]

    folder = [f for f in listdir('data/understat/')]
    ret_list = []

    us_data = {}
    for file in folder:
        if '201920' not in file:
            with open('data/understat/' + str(file)) as json_file:
                data = json.load(json_file)

            data = data['date']
            date_list = []
            home_list = []
            away_list = []
            home_rating = []
            away_rating = []

            for e in data:
                for match in e['home']:
                    home_list.append(match['name'])
                    home_rating.append(match['score'])
                for match in e['away']:
                    away_list.append(match['name'])
                    away_rating.append(match['score'])
                date_list.append(e['name'])
                ret_list.append([date_list, home_list, away_list, home_rating, away_rating])

                date_list = []
                home_list = []
                away_list = []
                home_rating = []
                away_rating = []

    return ret_list

def load_whoscored_data():
    # date: [home away homescore awayscore]

    folder = [f for f in listdir('data/whoscored/')]

    ret_list = []
    for file in folder:
        with open('data/whoscored/' + str(file), encoding='utf8') as json_file:
            data = json.load(json_file)

        data = data['details']

        date_list = []
        home_list = []
        away_list = []
        home_rating = []
        away_rating = []
        home_players = []
        away_players = []
        home_players_ratings = []
        away_players_ratings = []
        home_players_out = []
        home_players_in = []
        home_sub_time = []
        home_sub_in_names = []
        home_sub_ratings = []
        away_players_out = []
        away_players_in = []
        away_sub_time = []
        away_sub_in_names = []
        away_sub_ratings = []

        sub_out_list = []
        sub_in_list = []
        sub_time_list = []

        sub_in_list = []
        ratings_list = []

        counter = 0

        for e in data:
            date_list.append(e['date'])
            home_list.append(e['teams'][0]['name'])
            home_rating.append(e['teams'][0]['ratings'])
            away_list.append(e['teams'][1]['name'])
            away_rating.append(e['teams'][1]['ratings'])

            player_list = []
            rating_list = []
            for player in e['home_players']:
                player_list.append(player['name'])
                rating_list.append(player['ratings'])

            home_players.append(player_list)
            home_players_ratings.append(rating_list)

            player_list = []
            rating_list = []
            for player in e['away_players']:
                player_list.append(player['name'])
                rating_list.append(player['ratings'])

            away_players.append(player_list)
            away_players_ratings.append(rating_list)

            sub_in_list = []
            sub_out_list = []
            sub_time_list = []
            for player in e['subs_home']:
                try:
                    sub_in_list.append(player['player_in'])
                    sub_out_list.append(player['player_out'])
                    sub_time_list.append(player['sub_time'])
                except KeyError:
                    sub_in_list.append('')

            home_players_out.append(sub_out_list)
            home_players_in.append(sub_in_list)
            home_sub_time.append(sub_time_list)

            sub_in_list = []
            sub_out_list = []
            sub_time_list = []
            try:
                for player in e['subs_away']:
                        sub_in_list.append(player['player_in'])
                        sub_out_list.append(player['player_out'])
                        sub_time_list.append(player['sub_time'])
            except KeyError:
                pass

            away_players_out.append(sub_out_list)
            away_players_in.append(sub_in_list)
            away_sub_time.append(sub_time_list)

            sub_in_list = []
            sub_rating_list = []
            for player in e['home_sub_ratings']:
                sub_in_list.append(player['players'])
                sub_rating_list.append(player['name'])

            home_sub_in_names.append(sub_in_list)
            home_sub_ratings.append(sub_rating_list)

            sub_in_list = []
            sub_rating_list = []
            try:
                for player in e['away_sub_ratings']:
                    sub_in_list.append(player['players'])
                    sub_rating_list.append(player['name'])
            except KeyError:
                pass

            away_sub_in_names.append(sub_in_list)
            away_sub_ratings.append(sub_rating_list)



            ret_list.append([date_list[0], home_list[0], away_list[0], home_rating[0], away_rating[0], home_players[0], home_players_ratings[0],
                                  away_players[0], away_players_ratings[0], home_players_out[0], home_players_in[0], home_sub_time[0],
                                  home_sub_in_names[0], home_sub_ratings[0], away_players_out[0], away_players_in[0], away_sub_time[0],
                                  away_sub_in_names[0], away_sub_ratings[0]])
            date_list = []
            home_list = []
            away_list = []
            home_rating = []
            away_rating = []
            home_players = []
            away_players = []
            home_players_ratings = []
            away_players_ratings = []
            home_players_out = []
            home_players_in = []
            home_sub_time = []
            home_sub_in_names = []
            home_sub_ratings = []
            away_players_out = []
            away_players_in = []
            away_sub_time = []
            away_sub_in_names = []
            away_sub_ratings = []

    return ret_list

def correct_us_data(us_data):
    us_ = []
    for date in us_data:
        cur_date = date[0]
        for i, _ in enumerate(date[1]):
            us_.append([cur_date[0], date[1][i], date[2][i], date[3][i], date[4][i]])
    return us_

def date_correct(us, ws):
    us_data_corr = us
    ws_data_corr = ws

    for i, entry in enumerate(us):
        d = entry[0]
        d_ = (re.split(',' or ' ', d))
        d_ = d_[1:]
        d__ = re.split(' ', d_[0])[1:]

        year = d_[1]
        month = d__[0]
        day = d__[1]

        if month == 'January':
            month = '01'
        elif month == 'February':
            month = '02'
        elif month == 'March':
            month = '03'
        elif month == 'April':
            month = '04'
        elif month == 'May':
            month = '05'
        elif month == 'June':
            month = '06'
        elif month == 'July':
            month = '07'
        elif month == 'August':
            month = '08'
        elif month == 'September':
            month = '09'
        elif month == 'October':
            month = '10'
        elif month == 'November':
            month = '11'
        elif month == 'December':
            month = '12'

        us[i][0] = str(year)[3:] + str(month) + str(day)

    counter = 0
    for i, entry in enumerate(ws):
        d = entry[0]
        if 'Invalid date' in d:
            counter += 1
            year = 'inv' + str(counter)
            month = 'inv' + str(counter)
            day = ' inv' + str(counter)
        else:
            d_ = (re.split(',', d)[1])
            d__ = re.split('-', d_)

            year = d__[2]
            month = d__[1]
            day = d__[0]

        if month == 'Jan':
            month = '01'
        elif month == 'Feb':
            month = '02'
        elif month == 'Mac':
            month = '03'
        elif month == 'Apr':
            month = '04'
        elif month == 'Mei':
            month = '05'
        elif month == 'Jun':
            month = '06'
        elif month == 'Jul':
            month = '07'
        elif month == 'Ago':
            month = '08'
        elif month == 'Sep':
            month = '09'
        elif month == 'Okt':
            month = '10'
        elif month == 'Nov':
            month = '11'
        elif month == 'Des':
            month = '12'

        ws_data_corr[i][0] = year + month + day[1:]

    return us_data_corr, ws_data_corr



def check_only_us(us, ws):
    only_us = []
    for i, entry in enumerate(us):
        cur_date = entry[0]
        test = False
        for j, date in enumerate(ws):
            if cur_date == date[0]:
                test = True
        if test == False:
            only_us.append(cur_date)

def check_only_ws(us, ws):
    only_ws = []
    for i, entry in enumerate(ws):
        cur_date = entry[0]
        test = False
        for j, date in enumerate(us):
            if cur_date == date[0]:
                test = True
        if test == False:
            only_ws.append(cur_date)


def correct_team_names(us, ws):
    # id_list_us = []
    # for i, entry in enumerate(us):
    #     if entry[1] not in id_list_us:
    #         id_list_us.append(entry[1])
    # print(sorted(id_list_us))
    # id_list_ws = []
    # for i, entry in enumerate(ws):
    #     if entry[1] not in id_list_ws:
    #         id_list_ws.append(entry[1])
    # print(sorted(id_list_ws))
    #
    # for i, entry in enumerate(ws):
    #     team1 = entry[1]
    #     team2 = entry[2]

    for i, list in enumerate(ws):
        list[1:3] = [w.replace('Schalke', 'Schalke 04') for w in list[1:3]]
        list[1:3] = [w.replace('Bayern', 'Bayern Munich') for w in list[1:3]]
        list[1:3] = [w.replace('FC Koeln', 'FC Cologne') for w in list[1:3]]
        list[1:3] = [w.replace('Hamburg', 'Hamburger SV') for w in list[1:3]]
        list[1:3] = [w.replace('Hannover', 'Hannover 96') for w in list[1:3]]
        list[1:3] = [w.replace('Leverkusen', 'Bayer Leverkusen') for w in list[1:3]]
        list[1:3] = [w.replace('Mainz', 'Mainz 05') for w in list[1:3]]
        list[1:3] = [w.replace('Stuttgart', 'VfB Stuttgart') for w in list[1:3]]
        ws[i] = list

    return us, ws

def correct_date(us, ws):
    counterc = 0
    us_ = us
    ws_ = ws.copy()
    ws_ret = []
    for i in range(len(us)):
        counter = 0
        while True:
            if us[i][1:3] == ws[counter][1:3]:
                ws[counter][0] = us[i][0]
                ws_ret.append(ws[counter])
                del ws[counter]
                counterc += 1
                break
            counter = counter + 1
    print('Corrected ' + str(counterc))
    return us, ws_ret

def sort_that_shit(us, ws):
    sorted_us = []
    sorted_ws = []
    for i, entry in enumerate(us):
        us[i][0] = int(entry[0])
    for i, entry in enumerate(ws):
        ws[i][0] = int(entry[0])

    while True:
        max_us = 0
        max_ind = 0
        for i, entry in enumerate(us):
            if us[i][0] > max_us:
                max_us = us[i][0]
                max_ind = i
        sorted_us.append(us.pop(max_ind))
        if len(us) <= 0:
            break

    while True:
        max_ws = 0
        max_ind = 0
        for i, entry in enumerate(ws):
            if ws[i][0] > max_ws:
                max_ws = ws[i][0]
                max_ind = i
        sorted_ws.append(ws.pop(max_ind))
        if len(ws) <= 0:
            break

    return sorted_us, sorted_ws


def sort_that_shit_us(us):
    us[0] = us[0].tolist()
    game_id_list = []
    home_score = []
    away_score = []

    while True:
        max_us = 0
        max_ind = 0
        for i, entry in enumerate(us[0]):
            if entry > max_us:
                max_us = entry
                max_ind = i
        game_id_list.append(us[0].pop(max_ind))
        home_score.append(us[1][max_ind])
        away_score.append(us[2][max_ind])

        if len(us[0]) <= 0:
            break

    return [np.array(game_id_list), home_score, away_score]



def sort_that_shit_ws(ws):
    ws[0] = ws[0].tolist()
    game_id_list = []
    home_score = []
    away_score = []
    home_players = []
    away_players = []
    home_players_score = []
    away_players_score = []
    home_subs_time = []
    away_subs_time = []


    while True:
        max_ws = 0
        max_ind = 0
        for i, entry in enumerate(ws[0]):
            if entry > max_ws:
                max_ws = entry
                max_ind = i
        game_id_list.append(ws[0].pop(max_ind))
        home_score.append(ws[1][max_ind])
        away_score.append(ws[2][max_ind])
        home_players.append(ws[3][max_ind])
        home_players_score.append(ws[4][max_ind])
        home_subs_time.append(ws[5][max_ind])
        away_players.append(ws[6][max_ind])
        away_players_score.append(ws[7][max_ind])
        away_subs_time.append(ws[8][max_ind])

        if len(ws[0]) <= 0:
            break

    return [np.array(game_id_list), home_score, away_score, home_players, home_players_score,
            home_subs_time, away_players, away_players_score, away_subs_time]


def sort_that_shit_fd(fd):
    fd[0] = fd[0].tolist()
    game_id_list = []
    home_goals = []
    away_goals = []
    result = []
    half_time_goals_home = []
    half_time_goals_away = []
    half_time_result = []

    hshots = []
    ashots = []
    hst = []
    ast = []
    hf = []
    af = []
    hc = []
    ac = []
    hy = []
    ay = []
    hr = []
    ar = []

    b365h = []
    b365d = []
    b365a = []

    while True:
        max_fd = 0
        max_ind = 0
        for i, entry in enumerate(fd[0]):
            if entry > max_fd:
                max_fd = entry
                max_ind = i
        game_id_list.append(fd[0].pop(max_ind))
        home_goals.append(fd[1][max_ind])
        away_goals.append(fd[2][max_ind])
        result.append(fd[3][max_ind])
        half_time_goals_home.append(fd[4][max_ind])
        half_time_goals_away.append(fd[5][max_ind])
        half_time_result.append(fd[6][max_ind])

        hshots.append(fd[7][max_ind])
        ashots.append(fd[8][max_ind])
        hst.append(fd[9][max_ind])
        ast.append(fd[10][max_ind])
        hf.append(fd[11][max_ind])
        af.append(fd[12][max_ind])
        hc.append(fd[13][max_ind])
        ac.append(fd[14][max_ind])
        hy.append(fd[15][max_ind])
        ay.append(fd[16][max_ind])
        hr.append(fd[17][max_ind])
        ar.append(fd[18][max_ind])

        b365h.append(fd[19][max_ind])
        b365d.append(fd[20][max_ind])
        b365a.append(fd[21][max_ind])
        if len(fd[0]) <= 0:
            break

    return [np.array(game_id_list), home_goals, away_goals, result, half_time_goals_home, half_time_goals_away
        , half_time_result, hshots, ashots, hst, ast, hf, af, hc, ac, hy, ay, hr, ar,
            b365h, b365d, b365a]


def load_football_data():
    folder = [f for f in listdir('data/football_data/')]

    date_list = []
    home_list = []
    away_list = []
    home_goals = []
    away_goals = []
    result = []
    half_time_goals_home = []
    half_time_goals_away = []
    half_time_result = []

    hshots = []
    ashots = []
    hst = []
    ast = []
    hf = []
    af = []
    hc = []
    ac = []
    hy = []
    ay = []
    hr = []
    ar = []

    b365h = []
    b365d = []
    b365a = []

    for i, file in enumerate(folder):
        with open('data/football_data/' + str(file), 'r') as f:
            reader = csv.reader(f)
            data_array = list(reader)
        data_array = list(map(list, zip(*data_array)))

        for i in data_array[1][1:]:
            date_list.append(i)
        for i in data_array[2][1:]:
            home_list.append(i)
        for i in data_array[3][1:]:
            away_list.append(i)
        for i in data_array[4][1:]:
            home_goals.append(i)
        for i in data_array[5][1:]:
            away_goals.append(i)
        for i in data_array[6][1:]:
            result.append(i)
        for i in data_array[7][1:]:
            half_time_goals_home.append(i)
        for i in data_array[8][1:]:
            half_time_goals_away.append(i)
        for i in data_array[9][1:]:
            half_time_result.append(i)
        for i in data_array[10][1:]:
            hshots.append(i)
        for i in data_array[11][1:]:
            ashots.append(i)
        for i in data_array[12][1:]:
            hst.append(i)
        for i in data_array[13][1:]:
            ast.append(i)
        for i in data_array[14][1:]:
            hf.append(i)
        for i in data_array[15][1:]:
            af.append(i)
        for i in data_array[16][1:]:
            hc.append(i)
        for i in data_array[17][1:]:
            ac.append(i)
        for i in data_array[18][1:]:
            hy.append(i)
        for i in data_array[19][1:]:
            ay.append(i)
        for i in data_array[20][1:]:
            hr.append(i)
        for i in data_array[21][1:]:
            ar.append(i)
        for i in data_array[22][1:]:
            b365h.append(i)
        for i in data_array[23][1:]:
            b365d.append(i)
        for i in data_array[24][1:]:
            b365a.append(i)

    ret_list = [date_list, home_list, away_list, home_goals, away_goals, result, half_time_goals_home,
                         half_time_goals_away, half_time_result, hshots, ashots, hst, ast, hc, ac, hf, af, hy, ay, hr,
                         ar, b365h, b365d, b365a]

    return list(map(list, zip(*ret_list)))


def fd_correct_date(fd):
    fd = list(map(list, zip(*fd)))
    for i, date in enumerate(fd[0]):
        year = date[-2:]
        month = date[3:5]
        day = date[0:2]
        fd[0][i] = int(year + month + day)
    return fd

def correct_team_names_fd(fd):
    # fd[1] = [w.replace('Schalke', 'Schalke 04') for w in fd[1]]
    for i in [1, 2]:
        fd[i] = [w.replace('FC Koln', 'FC Cologne') for w in fd[i]]
        fd[i] = [w.replace('Hamburg', 'Hamburger SV') for w in fd[i]]
        fd[i] = [w.replace('Hannover', 'Hannover 96') for w in fd[i]]
        fd[i] = [w.replace('Leverkusen', 'Bayer Leverkusen') for w in fd[i]]
        fd[i] = [w.replace('Mainz', 'Mainz 05') for w in fd[i]]
        fd[i] = [w.replace('Stuttgart', 'VfB Stuttgart') for w in fd[i]]
        fd[i] = [w.replace('Dortmund', 'Borussia Dortmund') for w in fd[i]]
        fd[i] = [w.replace('Ein Frankfurt', 'Eintracht Frankfurt') for w in fd[i]]
        fd[i] = [w.replace('Fortuna Dusseldorf', 'Fortuna Duesseldorf') for w in fd[i]]
        fd[i] = [w.replace('Hertha', 'Hertha Berlin') for w in fd[i]]
        fd[i] = [w.replace("M'gladbach", 'Borussia M.Gladbach') for w in fd[i]]
        fd[i] = [w.replace("Nurnberg", 'Nuernberg') for w in fd[i]]
        fd[i] = [w.replace("RB Leipzig", 'RasenBallsport Leipzig') for w in fd[i]]
    return fd

def sort_us(us):
    date = []
    home = []
    home_score = []
    away = []
    away_score = []
    for ilist in us:
        date.append(ilist[0])
        home.append(ilist[1])
        away.append(ilist[2])
        home_score.append(ilist[3])
        away_score.append(ilist[4])
    return [date, home, away, home_score, away_score]


def sort_us(us):
    date = []
    home = []
    home_score = []
    away = []
    away_score = []
    for ilist in us:
        date.append(ilist[0])
        home.append(ilist[1])
        away.append(ilist[2])
        home_score.append(ilist[3])
        away_score.append(ilist[4])
    return [date, home, away, home_score, away_score]

def sort_ws(ws):
    date = []
    home = []
    home_score = []
    away = []
    away_score = []
    home_players = []
    away_players = []
    home_players_score = []
    away_players_score = []
    home_subs_out = []
    home_subs_in_tim = []
    home_subs_in_rat = []
    home_subs_time = []
    home_subs_rating = []
    away_subs_out = []
    away_subs_in_tim = []
    away_subs_in_rat = []
    away_subs_time = []
    away_subs_rating = []

    for ilist in ws:
        date.append(ilist[0])
        home.append(ilist[1])
        away.append(ilist[2])
        home_score.append(ilist[3])
        away_score.append(ilist[4])
        home_players.append(ilist[5])
        home_players_score.append(ilist[6])
        away_players.append(ilist[7])
        away_players_score.append(ilist[8])
        home_subs_out.append(ilist[9])
        home_subs_in_tim.append(ilist[10])
        home_subs_in_rat.append(ilist[12])
        home_subs_time.append(ilist[11])
        home_subs_rating.append(ilist[13])
        away_subs_out.append(ilist[14])
        away_subs_in_tim.append(ilist[15])
        away_subs_in_rat.append(ilist[17])
        away_subs_time.append(ilist[16])
        away_subs_rating.append(ilist[18])

    return [date, home, away, home_score, away_score, home_players, home_players_score, away_players, away_players_score,
            home_subs_out, home_subs_in_tim, home_subs_time, home_subs_in_rat, home_subs_rating, away_subs_out, away_subs_in_tim, away_subs_time,
            away_subs_in_rat, away_subs_rating]


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

def correct_sub_names(ws):
    home_tim = []
    away_tim = []

    for k, ilist in enumerate(ws[10]):
        ws[10][k] = [i for i in ilist if i]
        for j, i in enumerate(ilist):
            if hasNumbers(i):
                ws[10][k].pop(j)

    for k, ilist in enumerate(ws[15]):
        ws[15][k] = [i for i in ilist if i]
        for j, i in enumerate(ilist):
            if hasNumbers(i):
                ws[15][k].pop(j)

    for k, ilist in enumerate(ws[10]):
        for i, name in enumerate(ilist):
            try:
                ws[10][k][i] = re.split(' ', name)[-1]
            except IndexError:
                print('error')
    for k, ilist in enumerate(ws[15]):
        for i, name in enumerate(ilist):
            try:
                ws[15][k][i] = re.split(' ', name)[-1]
            except IndexError:
                print('error')
    print('Finished')
    return ws

def sort_calculate_players(ws):
    ret_home_players = []
    ret_home_players_scores = []
    ret_home_players_times = []
    ret_away_players = []
    ret_away_players_scores = []
    ret_away_players_times = []


    for ilist in ws:
        ret_home_players.append(ilist[0])
        ret_home_players_scores.append(ilist[1])
        ret_home_players_times.append(ilist[2])
        ret_away_players.append(ilist[3])
        ret_away_players_scores.append(ilist[4])
        ret_away_players_times.append(ilist[5])

    return [ret_home_players, ret_home_players_scores, ret_home_players_times,
            ret_away_players, ret_away_players_scores, ret_away_players_times]


def calculate_player_times(ws):
    ret = []

    home_players = ws[5]
    home_players_score = ws[6]
    away_players = ws[7]
    away_players_score = ws[8]
    player_sub = ws[-10:]
    home_subs_out = player_sub[0]
    home_subs_in_tim = player_sub[1]
    home_subs_time = player_sub[2]
    home_subs_in_rat = player_sub[3]
    home_subs_rating = player_sub[4]
    away_subs_out = player_sub[5]
    away_subs_in_tim = player_sub[6]
    away_subs_time = player_sub[7]
    away_subs_in_rat = player_sub[8]
    away_subs_rating = player_sub[9]

    for i in range(len(home_subs_out)):
        ret_home_players = []
        ret_home_players_times = []
        ret_home_players_scores = []
        for player in home_players[i]:
            if player not in home_subs_out[i]:
                ret_home_players.append(player)
                ret_home_players_times.append('90')
                ret_home_players_scores.append(home_players_score[i][home_players[i].index(player)])
            else:
                ret_home_players.append(player)
                try:
                    ret_home_players_times.append(home_subs_time[i][home_subs_out[i].index(player)])
                    ret_home_players_scores.append(home_players_score[i].index(player))
                except ValueError:
                    print(str(player) + ' missing!')
                    ret_home_players.pop(-1)
        for player in home_subs_in_rat[i]:
            ret_home_players.append(player)
            try:
                ret_home_players_times.append('90-' + str(home_subs_time[i][home_subs_in_tim[i].index(player)]))
                ret_home_players_scores.append(home_subs_rating[i][home_subs_in_rat[i].index(player)])
            except ValueError:
                print(str(player) + ' missing!')
                ret_home_players.pop(-1)

        ret_away_players = []
        ret_away_players_times = []
        ret_away_players_scores = []
        for player in away_players[i]:
            if player not in away_subs_out[i]:
                ret_away_players.append(player)
                ret_away_players_times.append('90')
                ret_away_players_scores.append(away_players_score[i][away_players[i].index(player)])
            else:
                ret_away_players.append(player)
                try:
                    ret_away_players_times.append(away_subs_time[i][away_subs_out[i].index(player)])
                    ret_away_players_scores.append(away_players_score[i].index(player))
                except ValueError:
                    print(str(player) + ' missing!')
                    ret_away_players.pop(-1)
        for player in away_subs_in_rat[i]:
            ret_away_players.append(player)
            try:
                ret_away_players_times.append('90-' + str(away_subs_time[i][away_subs_in_tim[i].index(player)]))
                ret_away_players_scores.append(away_subs_rating[i][away_subs_in_rat[i].index(player)])
            except ValueError:
                print(str(player) + ' missing!')
                ret_away_players.pop(-1)

        ret.append([ret_home_players, ret_home_players_scores, ret_home_players_times, ret_away_players,
                    ret_away_players_scores, ret_away_players_times])

    ret = sort_calculate_players(ret)

    for i in range(5, len(ws)):
        ws.pop(-1)
    for i in range(len(ret)):
        ws.append(ret[i])
    return ws

def calculate_player_times_2(ws):
    for k in [7, 10]:
        for i, ilist in enumerate(ws[k]):
            for j, time in enumerate(ilist):
                if "'" in time:
                    time = time[:-1]
                if "+" in time:
                    time = time[:-2]
                if "-" in time:
                    time1 = re.split('-', time)[0]
                    time2 = re.split('-', time)[1]
                    ws[k][i][j] = int(time1) - int(time2)
                else:
                    ws[k][i][j] = int(time)
    return ws


def flip_fd(fd):
    for i, category in enumerate(fd):
        fd[i] = np.flip(category, axis=0)
    return fd

def prepare_team_dict():
    team_dict = {}
    for i, team in enumerate(['Augsburg', 'Bayer Leverkusen', 'Bayern Munich', 'Borussia Dortmund',
                              'Borussia M.Gladbach', 'Darmstadt', 'Eintracht Frankfurt', 'FC Cologne',
                              'Fortuna Duesseldorf', 'Freiburg', 'Hamburger SV', 'Hannover 96',
                              'Hertha Berlin', 'Hoffenheim', 'Ingolstadt', 'Mainz 05', 'Nuernberg',
                              'Paderborn', 'RasenBallsport Leipzig', 'Schalke 04', 'VfB Stuttgart',
                              'Werder Bremen', 'Wolfsburg']):
        team_dict[team] = i + 1
    return team_dict


def generate_game_id(us, ws, fd):
    team_dict = prepare_team_dict()
    nb_games = len(us[0])
    # GAME ID
    game_id = np.zeros(nb_games)
    for i in range(nb_games):
        game_id[i] = 100000000000 * team_dict[us[1][i]] + 10000000 * team_dict[us[2][i]] + us[0][i]

    # GAME ID 2
    game_id_2 = np.zeros(nb_games)
    for i in range(nb_games):
        game_id_2[i] = 100000000000 * team_dict[ws[1][i]] + 10000000 * team_dict[ws[2][i]] + ws[0][i]

    # GAME ID 3
    game_id_3 = np.zeros(nb_games)
    for i in range(nb_games):
        game_id_3[i] = 100000000000 * team_dict[fd[1][i]] + 10000000 * team_dict[fd[2][i]] + fd[0][i]

    for i in range(2):
        us.pop(0)
        ws.pop(0)
        fd.pop(0)
    us[0] = game_id
    ws[0] = game_id_2
    fd[0] = game_id_3

    return us, ws, fd


def process_raw_data():
    us = correct_us_data(load_understat_data())
    ws = load_whoscored_data()
    us, ws = correct_team_names(us, ws)
    us, ws = date_correct(us, ws)
    us, ws = correct_date(us, ws)
    us, ws = sort_that_shit(us, ws)
    us = sort_us(us)
    ws = sort_ws(ws)
    ws = correct_sub_names(ws)
    ws = calculate_player_times(ws)
    ws = calculate_player_times_2(ws)
    fd = load_football_data()
    fd = fd_correct_date(fd)
    fd = correct_team_names_fd(fd)
    fd = flip_fd(fd)
    us, ws, fd = generate_game_id(us, ws, fd)
    us = sort_that_shit_us(us)
    ws = sort_that_shit_ws(ws)
    fd = sort_that_shit_fd(fd)
    return us, ws, fd


def get_base_data(us, ws, fd):
    team_dict = prepare_team_dict()
    nb_teams = len(team_dict)
    nb_games = len(us[0])
    data = np.zeros((81, 1530))

    game_id = np.zeros(nb_games)

    expg_home = np.zeros(nb_games)
    expg_away = np.zeros(nb_games)
    home_rat = np.zeros(nb_games)
    away_rat = np.zeros(nb_games)
    h_player_rat = np.zeros((14, nb_games))
    a_player_rat = np.zeros((14, nb_games))
    h_times = np.zeros((14, nb_games))
    a_times = np.zeros((14, nb_games))
    hg = np.zeros(nb_games)
    ag = np.zeros(nb_games)
    hw = np.zeros(nb_games)
    draw = np.zeros(nb_games)
    aw = np.zeros(nb_games)
    hshots = np.zeros(nb_games)
    ashots = np.zeros(nb_games)
    hst = np.zeros(nb_games)
    ast = np.zeros(nb_games)
    hf = np.zeros(nb_games)
    af = np.zeros(nb_games)
    hc = np.zeros(nb_games)
    ac = np.zeros(nb_games)
    hy = np.zeros(nb_games)
    ay = np.zeros(nb_games)
    hr = np.zeros(nb_games)
    ar = np.zeros(nb_games)

    b365h = np.zeros(nb_games)
    b365d = np.zeros(nb_games)
    b365a = np.zeros(nb_games)


    for i in range(nb_games):
        # Game Id
        game_id[i] = float(us[0][i])

        # expected goals
        expg_home[i] = float(us[-2][i])
        expg_away[i] = float(us[-1][i])

        # team ratings
        home_rat[i] = float(ws[1][i])
        away_rat[i] = float(ws[2][i])

        # player ratings
        for k, rating in enumerate(ws[4][i]):
            h_player_rat[k, i] = float(rating)
        for k, rating in enumerate(ws[7][i]):
            a_player_rat[k, i] = float(rating)

        # playing times
        for k, time in enumerate(ws[5][i]):
            h_times[k, i] = float(time)
        for k, time in enumerate(ws[8][i]):
            a_times[k, i] = float(time)

        # Goals
        hg[i] = float(fd[1][i])
        ag[i] = float(fd[2][i])

        # Full Time Result Result
        if fd[3][i] == 'H':
            hw[i] = 1.
            draw[i] = 0.
            aw[i] = 0.
        elif fd[3][i] == 'D':
            hw[i] = 0.
            draw[i] = 1.
            aw[i] = 0.
        elif fd[3][i] == 'A':
            hw[i] = 0.
            draw[i] = 0.
            aw[i] = 1.

        # Number shots
        hshots[i] = float(fd[7][i])
        ashots[i] = float(fd[8][i])

        # Shots on Target
        hst[i] = float(fd[9][i])
        ast[i] = float(fd[10][i])

        # Fouls
        hf[i] = float(fd[11][i])
        af[i] = float(fd[12][i])

        # Corners
        hc[i] = float(fd[13][i])
        ac[i] = float(fd[14][i])

        # Yellows
        hy[i] = float(fd[15][i])
        ay[i] = float(fd[16][i])

        # Reds
        hr[i] = float(fd[17][i])
        ar[i] = float(fd[18][i])

        # Odds
        b365h[i] = float(fd[19][i])
        b365d[i] = float(fd[20][i])
        b365a[i] = float(fd[21][i])

    data[0] = game_id
    data[1] = expg_home
    data[2] = expg_away
    data[3] = home_rat
    data[4] = away_rat
    data[5:19] = h_player_rat
    data[19:33] = a_player_rat
    data[33:47] = h_times
    data[47:61] = a_times
    data[61] = hg
    data[62] = ag
    data[63] = hw
    data[64] = draw
    data[65] = aw
    data[66] = hshots
    data[67] = ashots
    data[68] = hst
    data[69] = ast
    data[70] = hf
    data[71] = af
    data[72] = hc
    data[73] = ac
    data[74] = hy
    data[75] = ay
    data[76] = hr
    data[77] = ar
    data[78] = b365h
    data[79] = b365d
    data[80] = b365a

    return data

def get_away_id(game_ids):
    strings = []
    for i, id in enumerate(game_ids):
        strings.append(str(id)[3:])
    ret_game_ids = np.zeros_like(game_ids)
    for i, string in enumerate(strings):
        ret_game_ids[i] = float(string)
    return ret_game_ids

def create_team_data(data):
    team_dict = prepare_team_dict()
    nb_teams = len(team_dict)

    print(data.shape)

    team_list = []
    for i in range(1, nb_teams):
        cur_id_h = 100000000000 * i
        cur_id_a = 10000000 * i
        cur_games_ind = np.argwhere(np.abs(data[0] - cur_id_h) < 1000000000)
        mod_data = get_away_id(data[0])
        cur_games_ind_a = np.argwhere(np.abs(mod_data - cur_id_a) < 1000000)
        games_ind = np.concatenate([cur_games_ind, cur_games_ind_a], axis=0)
        games = np.zeros((81, games_ind.shape[0]))
        games = data[:, games_ind][:, :, 0]
        team_list.append(games)

    return np.concatenate(team_list, axis=1)

def prepare_short_data(data):
    home = np.zeros(data.shape[1])
    away = np.zeros(data.shape[1])
    mod_id = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        cur_id = str(data[0, i])
        cur_home = int(cur_id[:-13])
        cur_away = int(cur_id[-11:-9])
        home[i] = cur_home
        away[i] = cur_away
        mod_id[i] = int(str(cur_id)[-8:-2])
    mod_id = np.expand_dims(mod_id, axis=0)
    home = np.expand_dims(home, axis=0)
    away = np.expand_dims(away, axis=0)
    ret = np.concatenate([mod_id, home, away, data[1:, :]], axis=0)
    return ret


def get_last_games(data, date, home, away):
    home_data_h = np.argwhere(data[1, :] == home)[:, 0]
    home_data_a = np.argwhere(data[2, :] == home)[:, 0]

    away_data_h = np.argwhere(data[1, :] == away)[:, 0]
    away_data_a = np.argwhere(data[2, :] == away)[:, 0]

    home_data_h_exp = np.zeros((data.shape[0] + 2, home_data_h.shape[0]))
    home_data_a_exp = np.zeros((data.shape[0] + 2, home_data_a.shape[0]))
    away_data_h_exp = np.zeros((data.shape[0] + 2, away_data_h.shape[0]))
    away_data_a_exp = np.zeros((data.shape[0] + 2, away_data_a.shape[0]))

    home_data_h_exp = data[:, np.argwhere(data[0, home_data_h] < date)[:, 0]]
    # set team id to 1 and enemy team to 0
    home_data_h_exp[-2, :] = np.ones_like(home_data_h_exp[1, :])
    home_data_h_exp[-1, :] = np.zeros_like(home_data_h_exp[2, :])

    home_data_a_exp = data[:, np.argwhere(data[0, away_data_a] < date)[:, 0]]
    # set team id to 1 and enemy team to 0
    home_data_a_exp[-1, :] = np.ones_like(home_data_a_exp[2, :])
    home_data_a_exp[-2, :] = np.zeros_like(home_data_a_exp[1, :])

    away_data_h_exp = data[:, np.argwhere(data[0, away_data_h] < date)[:, 0]]
    # set team id to 1 and enemy team to 0
    away_data_h_exp[-2, :] = np.ones_like(away_data_h_exp[1, :])
    away_data_h_exp[-1, :] = np.zeros_like(away_data_h_exp[2, :])

    away_data_a_exp = data[:, np.argwhere(data[0, away_data_a] < date)[:, 0]]
    # set team id to 1 and enemy team to 0
    away_data_a_exp[-1, :] = np.ones_like(away_data_a_exp[2, :])
    away_data_a_exp[-2, :] = np.zeros_like(away_data_a_exp[1, :])

    home_data = np.concatenate([home_data_h_exp, home_data_a_exp], axis=1)
    away_data = np.concatenate([away_data_h_exp, away_data_a_exp], axis=1)

    sorted_hd = home_data[:, np.flip(np.argsort(home_data[0, :]))]
    sorted_ad = away_data[:, np.flip(np.argsort(away_data[0, :]))]

    return sorted_hd, sorted_ad

def get_team_rat_10(data, date, team):

    data_red = data[0:7, :]

    team_h = np.argwhere(data_red[1, :] == team)[:, 0]
    team_a = np.argwhere(data_red[2, :] == team)[:, 0]

    team_data_h = data_red[:, np.argwhere(data_red[0, team_h] < date)[:, 0]]
    team_data_a = data_red[:, np.argwhere(data_red[0, team_a] < date)[:, 0]]

    conc = np.concatenate([team_data_h[[3, 5], :], team_data_a[[4, 6], :]], axis=1)

    if conc.size == 0:
        return np.array([0., 0.])
    else:
        rat = np.mean(conc, axis=1)
        return rat

def get_train_data(data, date, home, away):
    home_games, away_games = get_last_games(data, date, home, away)

    hg_ten = home_games[:, :10]
    ag_ten = away_games[:, :10]
    hg_ten_rated = np.zeros((hg_ten.shape[0] + 4, hg_ten.shape[1]))
    ag_ten_rated = np.zeros((ag_ten.shape[0] + 4, ag_ten.shape[1]))
    for i in range(10):
        hg_ten_rated[:, i] = np.concatenate([hg_ten[:, i], get_team_rat_10(data, hg_ten[0, i], hg_ten[1, i]),
                                       get_team_rat_10(data, hg_ten[0, i], hg_ten[2, i])], axis=0)
        ag_ten_rated[:, i] = np.concatenate([ag_ten[:, i], get_team_rat_10(data, ag_ten[0, i], ag_ten[1, i]),
                                       get_team_rat_10(data, ag_ten[0, i], ag_ten[2, i])], axis=0)
    return hg_ten_rated, get_team_rat_10(data, hg_ten_rated[0, -1], team=home), \
           ag_ten_rated, get_team_rat_10(data, ag_ten_rated[0, -1], team=away)

# us, ws, fd = d.process_raw_data()
# data = d.get_base_data(us, ws, fd)
# tdata = d.create_team_data(data)