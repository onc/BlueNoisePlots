#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import seaborn as sns
import pandas as pd
import numpy as np
from gapminder import gapminder
import matplotlib

sys.path.append('..')
from blue_noise_plot import jitter, blue_noise

IMAGE_FOLDER = Path('example_plots')
FILE_TYPE = '.png'

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

sns.set_style('whitegrid', {
    'axes.linewidth': 1,
    'axes.edgecolor': 'none'
})

font = {'family': 'Times New Roman', 'size': 15}
matplotlib.rc('font', **font)

def example_plots(orient, size, num_example):
    paper_plots_folder = IMAGE_FOLDER

    def singleclass(orient, size, rep=''):
        # =========================================================================================
        print('SINGLE CLASS')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'single_class'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_single_class_datasets = [
            { 'name': 'geyser', 'x': 'duration' },
            { 'name': 'tips', 'x': 'total_bill' },
            { 'name': 'iris', 'x': 'petal_length' },
        ]

        for dataset in seaborn_single_class_datasets:
            data = sns.load_dataset(dataset['name'])
            filename = dataset['name'] + '_' + dataset['x'] + '_' + str(rep)

            num_entries = int(data.shape[0])
            data = data.sample(n=int(num_entries * 0.6), replace=False)
            data = data.dropna(subset=[dataset['x']])

            if dataset['name'] == 'titanic' and dataset['x'] == 'fare':
                data = data[data['fare'] < 500]

            # jitter(x=dataset['x'], data=data, orient=orient, size=size,
            #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
            blue_noise(x=dataset['x'], data=data, orient=orient, size=size,
                       filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
            blue_noise(x=dataset['x'], data=data, orient=orient, size=size, centralized=True,
                       filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

        print('local datasets')
        print('total sleep')
        sleep_filename = 'csv_data/sleep.csv'
        sleep = pd.read_csv(sleep_filename)
        sleep = sleep.replace(r'^\s*$', np.nan, regex=True)
        sleep = sleep.dropna(subset=['total sleep'])
        sleep['total sleep'] = pd.to_numeric(sleep['total sleep'], downcast="float")

        filename = 'sleep_total_sleep_jitter' + str(rep)
        # jitter(x='total sleep', data=sleep, orient=orient, size=size,
        #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
        blue_noise(x='total sleep', data=sleep, orient=orient, size=size,
                   filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
        blue_noise(x='total sleep', data=sleep, orient=orient, size=size, centralized=True,
                   filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

    def multiclass(orient, size, rep=''):
        # =========================================================================================
        print('MULTICLASS')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'multi_class'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_multi_class_datasets = [
            { 'name': 'penguins', 'x': 'bill_length_mm', 'hue': 'sex' },
        ]

        for dataset in seaborn_multi_class_datasets:
            data = sns.load_dataset(dataset['name'])
            filename = dataset['name'] + '_' + dataset['x'] + '_' + str(rep)

            num_entries = int(data.shape[0] * 0.4)
            data = data.sample(n=num_entries, replace=False)
            data = data.dropna(subset=[dataset['x']])

            if dataset['name'] == 'titanic' and dataset['x'] == 'fare':
                data = data[data['fare'] < 500]

            filename = dataset['name'] + '_' + dataset['x'] + '_' + dataset['hue'] + '_' + str(rep)
            # jitter(x=dataset['x'], hue=dataset['hue'], data=data, orient=orient, size=size,
            #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
            blue_noise(x=dataset['x'], hue=dataset['hue'], data=data, orient=orient, size=size,
                       filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
            blue_noise(x=dataset['x'], hue=dataset['hue'], data=data, orient=orient, size=size,
                       centralized=True, filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

        # COVID
        print('covid')
        data_filename = 'csv_data/covid.csv'
        col_names = ['cat', 'X', 'Y']
        covid = pd.read_csv(data_filename, names=col_names)

        subset = covid[covid['cat'].isin(['A', 'B', 'D'])]

        filename = 'covid_viralload_' + str(rep)
        # jitter(x='Y', hue='cat', data=subset, orient=orient, size=size,
        #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
        blue_noise(x='Y', hue='cat', data=subset, orient=orient, size=size,
                   filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
        blue_noise(x='Y', hue='cat', data=subset, orient=orient, size=size, centralized=True,
                   filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

        subset = gapminder.sample(n=150, replace=False)
        print('gapminder')

        filename = 'gapminder_exp_continent_' + str(rep)
        # jitter(x='lifeExp', hue='continent', data=subset,
        #        orient=orient, size=size,
        #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
        blue_noise(x='lifeExp', hue='continent', data=subset,
                   orient=orient, size=size,
                   filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
        blue_noise(x='lifeExp', hue='continent', data=subset,
                   orient=orient, size=size, centralized=True,
                   filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))


    def multiclass_dodge(orient, size, rep=''):
        # =========================================================================================
        print('MULTICLASS DODGE')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'multi_class_dodge'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_multi_class_datasets = [
            { 'name': 'tips', 'x': 'total_bill', 'hue': 'time' },
        ]

        for dataset in seaborn_multi_class_datasets:
            data = sns.load_dataset(dataset['name'])

            num_entries = int(data.shape[0] * 0.4)
            data = data.sample(n=num_entries, replace=False)
            data = data.dropna(subset=[dataset['x']])

            filename = dataset['name'] + '_' + dataset['x'] + '_' + dataset['hue'] + '_' + str(rep)
            blue_noise(x=dataset['x'], hue=dataset['hue'], data=data, orient=orient, size=size, dodge=True,
                       filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))

            filename = dataset['name'] + '_' + dataset['x'] + '_' + dataset['hue'] + '_cent_' + str(rep)
            blue_noise(x=dataset['x'], hue=dataset['hue'], data=data, orient=orient, size=size,
                       dodge=True, centralized=True,
                       filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

    def quantizized(orient, size, rep=''):
        # =========================================================================================
        print('QUANTIZIZED')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'quantizized'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_multi_class_datasets = [
            { 'name': 'iris', 'x': 'petal_length' },
           ]

        for dataset in seaborn_multi_class_datasets:
            data = sns.load_dataset(dataset['name'])
            filename = dataset['name'] + '_' + dataset['x'] + '_' + str(rep)

            num_entries = int(data.shape[0] * 0.8)
            data = data.sample(n=num_entries, replace=False)
            data = data.dropna(subset=[dataset['x']])
            # jitter(x=dataset['x'], data=data, orient=orient, size=size,
            #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
            blue_noise(x=dataset['x'], data=data, orient=orient, size=size,
                       filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))

        # MPG quantizized!
        print('mpg')
        mpg_filename = 'csv_data/auto-mpg.data'
        column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                        'acceleration', 'year', 'origin', 'name']
        mpg = pd.read_csv(mpg_filename, delim_whitespace=True, names=column_names)

        subset = mpg[mpg['cylinders'].isin([4, 6, 8])]
        num_entries = int(subset.shape[0] * 0.5)
        subset = subset.sample(n=num_entries, replace=False)

        filename = 'cars_mpg_cylinders_' + str(rep)
        # jitter(x='mpg', hue='cylinders', data=subset, orient=orient, size=size,
        #        filename=sub_folder / str(filename + '_jitter_' + FILE_TYPE))
        blue_noise(x='mpg', hue='cylinders', data=subset, orient=orient, size=size,
                   filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
        blue_noise(x='mpg', hue='cylinders', data=subset, orient=orient, size=size, centralized=True,
                   filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

    def centralized(orient, size, rep=''):
        # =========================================================================================
        print('CENTRALIZED')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'centralized'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_single_class_datasets = [
            { 'name': 'geyser', 'x': 'duration' },
            { 'name': 'tips', 'x': 'total_bill' },
        ]

        for dataset in seaborn_single_class_datasets:
            data = sns.load_dataset(dataset['name'])
            filename = dataset['name'] + '_' + dataset['x'] + '_' + str(rep)

            num_entries = int(data.shape[0] * 0.4)
            data = data.sample(n=num_entries, replace=False)
            data = data.dropna(subset=[dataset['x']])

            if dataset['name'] == 'titanic' and dataset['x'] == 'fare':
                data = data[data['fare'] < 500]

            blue_noise(x=dataset['x'], data=data, orient=orient, size=size,
                       filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
            blue_noise(x=dataset['x'], data=data, orient=orient, size=size, centralized=True,
                       filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

    def num_points(orient, size, rep=''):
        # =========================================================================================
        print('COMPARE NUMBER OF POINTS')
        # =========================================================================================
        sub_folder = paper_plots_folder / 'num_points'
        Path(sub_folder).mkdir(parents=True, exist_ok=True)

        seaborn_single_class_datasets = [
            { 'name': 'geyser', 'x': 'duration' },
        ]

        for dataset in seaborn_single_class_datasets:
            data = sns.load_dataset(dataset['name'])
            data = data.dropna(subset=[dataset['x']])

            for sample_size in [64, 128, 256]:
                sub = data.sample(n=sample_size, replace=False)
                filename = str(sample_size) + '_' + dataset['name'] + '_' + dataset['x'] + '_' + str(rep)

                blue_noise(x=dataset['x'], data=sub, orient=orient, size=size,
                           filename=sub_folder / str(filename + '_bn_' + FILE_TYPE))
                blue_noise(x=dataset['x'], data=sub, orient=orient, size=size, centralized=True,
                           filename=sub_folder / str(filename + '_cent_' + FILE_TYPE))

    def plot(i):
        singleclass(orient, size, i)
        quantizized(orient, size, i)
        centralized(orient, size, i)
        multiclass(orient, size, i)
        multiclass_dodge(orient, size, i)
        num_points(orient, size, i)

    mpg_filename = 'csv_data/auto-mpg.data'
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                    'acceleration', 'year', 'origin', 'name']
    mpg = pd.read_csv(mpg_filename, delim_whitespace=True, names=column_names)
    subset = mpg[mpg['cylinders'].isin([4, 6, 8])]

    # Blue noise plot of `subset`, using our automatic width computation
    points = blue_noise(x='mpg', data=subset, orient='h')
    print('Num Classes: ', len(points))
    print('Num Points: ', len(points[0]))

    points = blue_noise(x='mpg', hue='cylinders', data=subset, orient='h')
    print('Num Classes: ', len(points))
    print('Num Points: ', len(points[0]))
    print('Num Points: ', len(points[1]))
    print('Num Points: ', len(points[2]))

    # Blue noise plot of `subset`, using predefined width.
    points = blue_noise(x='mpg', data=subset, orient='h', plot_width=0.3)

    # basic plot without rendering
    dataset = { 'name': 'geyser', 'x': 'duration' }
    data = sns.load_dataset(dataset['name'])
    data = data.dropna(subset=[dataset['x']])
    sub = data.sample(n=256, replace=False)
    points = blue_noise(x=dataset['x'], data=sub, orient=orient)
    print('num classes: ', len(points))

    blue_noise(x=dataset['x'], data=sub, filename='geyser-duration.png', orient=orient)
    blue_noise(x=dataset['x'], data=sub, plot_width=0.7, filename='geyser-duration-high.png', orient=orient)

    dataset = { 'name': 'penguins', 'x': 'bill_length_mm', 'hue': 'sex' }
    data = sns.load_dataset(dataset['name'])
    data = data.dropna(subset=[dataset['x']])
    sub = data.sample(n=256, replace=False)
    points = blue_noise(x=dataset['x'], hue=dataset['hue'], data=sub, orient=orient)
    print('num classes: ', len(points))

    plot(num_example)

def main():
    orient = 'h'
    marker_size = 20

    for n in range(1):
        print(n)
        example_plots(orient, marker_size, n)

if __name__ == "__main__":
    main()
