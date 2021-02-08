# Blue Noise Plots

Python implementation of *Blue Noise Plots*, a novel replacement for jitter plots, published at Eurographics 2021.

[C. van Onzenoodt](https://www.uni-ulm.de/en/in/mi/institute/staff/christian-van-onzenoodt/), [Gurprit Singh](https://people.mpi-inf.mpg.de/~gsingh/), [T. Ropinski](https://www.uni-ulm.de/in/mi/institut/mitarbeiter/timo-ropinski/), [T. Ritschel](http://www.homepages.ucl.ac.uk/~ucactri/), 

![Teaser](https://raw.githubusercontent.com/onc/BlueNoisePlots/master/images/Teaser.png)

[Paper Preprint]() [Project Page]()

## Prerequisites

* Python 3+

### Install dependencies

```
cd /BlueNoisePlots
pip install -r requirements.txt
```

## Usage

```python
from blue_noise_plot import blue_noise

# Load data
mpg_filename = 'csv_data/auto-mpg.data'
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'year', 'origin', 'name']
mpg = pd.read_csv(mpg_filename, delim_whitespace=True, names=column_names)
subset = mpg[mpg['cylinders'].isin([4, 6, 8])]

# Draw
blue_noise(x='mpg', hue='cylinders', data=subset, orient='h', size=20, filename=mpg-blue_noise_plot.png')
```

## Examples

![Blue Noise Plot examples](https://raw.githubusercontent.com/onc/BlueNoisePlots/master/images/Examples.png)
