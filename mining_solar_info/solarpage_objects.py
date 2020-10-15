import requests


class SolarPage:
    def __init__(self, latitude, longitude):
        self._url_api = 'https://re.jrc.ec.europa.eu/api/tmy?'
        self.content = None
        self._visit( self._url(latitude, longitude) )

        self.lat = float(latitude)
        self.lon = float(longitude)

    def _url(self, latitude, longitude):
        url = f'{self._url_api}lat={latitude}&lon={longitude}&time=local'
        return url

    def _visit(self, url):
        response = requests.get(url)
        response.encoding = 'utf-8'
        response.raise_for_status()

        self.content = response.text

    @staticmethod
    def headers():
        header1 = ['Date', 'Month', 'Day', 'NumDay', 'Hour', 'Lat', 'Lon', 'Alt']
        header2 = ['SP', 'RH', 'WS10m', 'WD10m', 'T2m', 'G(h)', 'Gb(n)', 'Gd(h)', 'IR(h)']
        headers = header1 + header2	

        return headers

        

