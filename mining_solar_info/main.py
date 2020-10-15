from manage_object import _new_scraper
from locations import locations


if __name__ == '__main__':
    locations = locations()
    
    for loc in locations:
        _new_scraper(loc['lat'], loc['lon'])
    