def majoraxis(img):
    """
    Find the major axis of the hand. The input image is a binary image of the hand mask.
    Return the angle and the y-intercept of the major axis. 
    Major axis is compute by minimizing the integral of the distances from all hand pixels to the 
    major axis. 


    @param np.array img     The input image A numpy.array of size (480,640), dtype=np._bool
    return (theta, b)       The majoraxis described with the angle theta and the y-intercept b. 
                            Theta is measured in degrees. """
    pass