# ML Model for Solar Radiation Prediction.

### About

This work is a effort for develop a machine learnig model for predict the solar radiation.

In all solar industries the correct prediction of solar radiation is a key on design and operation
proccess. Due the stochastic nature of solar radiation the existing empirical and mechanistic models
are good aproximations but with considerable errors.

The model developed in this work was trained with a dataset with meteorological and geographical data
of all country (Mexico). The data was obtained from [PV-GIS](https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html#PVP) trhough a scraper method and proccesed with 
the library [PV-LIB](https://github.com/pvlib/pvlib-python) developed by [Sandia National Laboratories](https://www.sandia.gov/)


### Sections

- `mining_solar_info`
  - This folder contains all logic for scraping the data and save the resulted datasets.

- `model`
  - `model/feature-engineering`: Is a collection of notebooks with the feature engineering proccess.
  -  `model/model-selection`: Is a a csollection of notebooks with the model selection procsess.
  - `model/ghi`: This folder contains all logic for tunning all best models, compare the result and generate plots and metadata of each result.

### Dataset
[Link to Kaggle](https://www.kaggle.com/emanuelosorio/datasets?campaign=50c1a4cf-95ba-4b4c-9c0c-2386bde2d81e)

### Results

The resultant model with the best performance was a:

**XGBR** (Xtreme Gradient Boosting Regretion)


###### Test Location

- SantaAna (Sonora Mexico)

###### Training Results:

- R2 = 0.9467821940386125
- RMSE = 76.9138282703692
- Training time: 7.2580 seconds

###### Test Results: 
- R2 = 0.9354960908553472
- RMSE = 83.21137350309252

##### Plots

* Real vs Predicted
![comparation_SantaAna_XGBR](https://user-images.githubusercontent.com/62397465/92851927-c4857e80-f3b3-11ea-8b5c-e20ba14bd285.jpg)

* Scatter Plot
![scattering_r2_XGBR_SantaAna](https://user-images.githubusercontent.com/62397465/92852105-ee3ea580-f3b3-11ea-8d3f-fe7379f24a0a.jpg)
