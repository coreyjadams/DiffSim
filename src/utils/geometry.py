import numpy


class GeometryUtils:

    def __init__(self, sipm_db):

        # Take the sipm db and use it to populate most information:

        # Min and max coordinates of sipm centers:
        self.min = [numpy.min(sipm_db.X), numpy.min(sipm_db.Y)]
        self.max = [numpy.max(sipm_db.X), numpy.max(sipm_db.Y)]

        # Spacing in x and y:
        self.spacing = [10., 10.]

    def xy_to_sipm_coordinate(self, xy)
        """
        Convert x/y location pairs to the nearest sipm coordinate.
        This isn't the index of the sipm, but rather an x/y integer
        pair.  For example, for a 15x15 grid of sipms spaced 10mm apart,
        with the origin in the center (sipm [7,7]), the x/y coordinates 
        [50.5, -17.3] (in mm) would end up at sipm coordinates as follows:

        sipm coordinate origin is (0,0) = (-70, -70) in proper coordinates.

        [50.5, -17.3] - [-70, -70] = [120.5, 52.7]

        [120.3, 52.7] / 10mm spacing = [12.3, 5.27]

        round down (truncate) to coordinates [12, 5]
        

        :param      x:         { parameter_description }
        :type       x:         { type_description }
        :param      y:         { parameter_description }
        :type       y:         { type_description }
        :param      database:  The database
        :type       database:  { type_description }
        """

        # Subtract off the minimum:

        result = (xy - self.min) / self.spacing

        # Cast the result to integer values:
        return result.astype(numpy.int64)


def nearest_sipm_index(x, y, database):
    pass

def sipm_to_xy(sipm_index, database):
    """
    Convert by sipm index to the central x/y coordinate
    
    :param      sipm_index:  The sipm index
    :type       sipm_index:  { type_description }
    :param      database:    The database
    :type       database:    { type_description }
    """


