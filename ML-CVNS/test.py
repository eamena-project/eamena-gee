print("hello world")
def upload_shapefile_to_gee(user, shp_file):
    """
    Upload a shapefile to Google Earth Engine as an asset.

    Args:
        user (django.contrib.auth.User): the request user.
        shp_file (shapefile.Reader): A shapefile reader object.
    """
    print(user.username)
    print(shp_file)