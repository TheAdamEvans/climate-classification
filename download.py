from subprocess import call
import sys

def download_year_data(year):
    """
    one year of hourly data from NOAA
    """
    url = [f'https://www.ncei.noaa.gov/data/global-hourly/archive/{year}.tar.gz']
    tarball = f'{year}.tar.gz'

    # download from NOAA
    call(["wget"]+url+['-O']+[tarball]+['-c'])

    # # uncompress
    # call(['mkdir',f'./data/{year}'])
    # call(['tar','-xzf',tarball,'-C',f'./data/{year}'])

    # # clean up
    # call(['rm',tarball])


if len(sys.argv) < 2:
    for year in range(1973, 2000):
        download_year_data(year)
else:
    download_year_data(sys.argv[1])
