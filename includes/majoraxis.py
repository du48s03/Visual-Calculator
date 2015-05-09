def majoraxis(mask):
    """
    Find the major axis of the hand. The input image is a binary image of the hand mask.
    Return the angle and the y-intercept of the major axis. 
    Major axis is compute by minimizing the integral of the distances from all hand pixels to the 
    major axis. 


    @param np.array img     The input image A numpy.array of size (480,640), dtype=np._bool
    return (theta, c_x, c_y) The majoraxis described with the angle theta and the center point of the object. 
                            Theta is measured in radius. """
    # non zero area
    (x,y) = mask.nonzero()
    # area
    area = len(x)
    # center position
    c_x = x.mean(axis=0)
    c_y = y.mean(axis=0)
    # second moment
    a_o = (np.multiply(x,x)).sum(axis=0)
    b_o = (np.multiply(x,y)).sum(axis=0)
    c_o = (np.multiply(y,y)).sum(axis=0)
    # the minimum moment of inerita
    # convert the second moment for the origin to the second moment for the center
    a = a_o - area*(c_x**2)
    b = 2*b_o - 2 * area * c_x * c_y
    c = c_o - area*(c_y**2)
    # find theta for the major axis
    theta = math.atan2(b,a-c)/2.0
    #print i,",",theta, ",",c_x,",",c_y
    return theta, c_x, c_y
