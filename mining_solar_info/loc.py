import pandas as pd
import locations

loc = locations.locations()

df = pd.DataFrame(loc)
filename = 'locations.csv'
df.to_csv(filename)
