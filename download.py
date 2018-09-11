from subprocess import call
from pathlib import Path
import sys

def download_year_data(year, path = Path('./data')):
    """
    one year of hourly data from NOAA
    """
    url = f'https://www.ncei.noaa.gov/data/global-hourly/archive/{year}.tar.gz'
    tarball = f'{year}.tar.gz'

    (path/'downloads/').mkdir(parents=True, exist_ok=True)

    # download from NOAA
    call(["wget"]+[url]+['-O']+[tarball]+['-c'])
    call(['mv']+[tarball]+[(path/'downloads/.')])

    # uncompress
    # call(['mkdir',f'./data/{year}'])
    (path/f'{year}').mkdir(parents=True, exist_ok=True)
    call(['tar','-xzf',(path/'downloads'/tarball),'-C',(path/f'{year}')])

    # clean up
    # call(['rm',tarball])


if len(sys.argv) < 2:
    for year in range(1973, 1980):
        download_year_data(year)
else:
    download_year_data(sys.argv[1])
