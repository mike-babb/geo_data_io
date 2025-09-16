#############################
#
# Compute the distance between two points and the compass bearing
#
#############################

# standard libraries
import math



def points2distance(start, end, unit):
    """
    Calculate distance (in kilometers) between two points given as (long, latt) pairs
    based on Haversine formula (http://en.wikipedia.org/wiki/Haversine_formula).
    Implementation inspired by JavaScript implementation from
    http://www.movable-type.co.uk/scripts/latlong.html
    Accepts coordinates as tuples (deg, min, sec), but coordinates can be given
    in any form - e.g. can specify only minutes:
    (0, 3133.9333, 0)
    is interpreted as
    (52.0, 13.0, 55.998000000008687)
    """
    # earths_radius_km = 6371 # kilometers
    # earths_radius_miles = 3959 # miles
    start_long = math.radians(start[0])
    start_latt = math.radians(start[1])
    end_long = math.radians(end[0])
    end_latt = math.radians(end[1])
    d_latt = end_latt - start_latt
    d_long = end_long - start_long
    a = math.sin(d_latt/2)**2 + math.cos(start_latt) * math.cos(end_latt) * math.sin(d_long/2)**2
    c = 2 * math.atan2(math.sqrt(a),  math.sqrt(1-a))

    if unit=="miles":
        earths_radius = 3959
    else:
        earths_radius = 6731
    # return the distance between the two points
    return earths_radius * c

def calculate_initial_compass_bearing(point_a, point_b):
    """
    Courtesy of: https://gist.github.com/jeromer/2005586
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - point_a: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - point_b: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(point_a) != tuple) or (type(point_b) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(point_a[1])
    lat2 = math.radians(point_b[1])

    diffLong = math.radians(point_b[0] - point_a[0])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

if __name__ == '__main__':


    start = (0,0)
    end = (-1,1)
    #outcome = points2distance(start, end, unit='miles')
    #print(outcome)

    outcome = calculate_initial_compass_bearing(point_a=start,
    point_b=end)
    print(outcome)


