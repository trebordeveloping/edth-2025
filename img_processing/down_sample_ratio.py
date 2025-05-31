
def calculate_gsd(alt: float, focal_length: float, pixel_size: float = 2.4*10**-3) -> float:
    """
    Calculate the Ground Sample Distance (GSD) in meters per pixel.

    :param alt: Altitude in meters.
    :param pixel_size: Size of a pixel in mm (default is 2.4 mm).
    :param focal_length: Focal length of the camera in millimeters.
    :return: GSD in meters per pixel.
    """
    return (alt * pixel_size) / focal_length

def calculate_down_sample_ratio(high_res_gsd: float, low_res_gsd: float) -> float:
    """
    Calculate the down sample ratio based on GSD values.

    :param high_res_gsd: GSD of the high resolution image in meters per pixel.
    :param low_res_gsd: GSD of the low resolution image in meters per pixel.
    :return: Down sample ratio.
    """
    return high_res_gsd / low_res_gsd