# ML-CVNS
> Machine Learning (ML) to automate detection of caravanserais (CVNS)

## Bases

### Learning base

- 60/80% of data tagged CVNS (examples)
- 40/20% of data tagged ⌐ CVNS (counter examples)
- 200/300 images
- 1 CVNS = 1 image
- image multispectral, multiscalar
- composed of images only, preferably in RAW (.gis) format
- number of pixels per image relatively the same from one image to another
- leave buffers (e.g. 10 m, 50 m)

#### Extraction of CVNS spatial prints
> *draw a polygon and extract the landcover inside, extract an image from Google Earth Engine, Access the underlying image*

The idea is to extract CVNS spatial prints from Google Earth Engine

![my caption](https://raw.githubusercontent.com/eamena-project/eamena-arches-dev/master/www/gee-cvns.png)
